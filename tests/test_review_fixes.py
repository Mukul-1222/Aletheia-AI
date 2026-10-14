from pathlib import Path
import torch
import json

from aletheia.config import load_config
from aletheia.agent import AletheiaAgent
from aletheia.curriculum import Lesson, iter_text_files
from aletheia.evaluation import best_byte_baseline_bpb, byte_unigram_bpb, tuned_byte_ngram_bpb
from aletheia.web import WebExplorer, WebConfig
import pytest


def small_cfg(tmp_path):
    cfg = load_config(Path('config.json'))
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    corpus = (
        "What did you learn today? How are you when you meet another person, and what do you notice in an ordinary conversation? "
        "A person can ask for clarification when a sentence is unclear, and a listener can use previous context to understand what the speaker means.\n"
        "A calm river moved through fields while distant bells marked the hour. People gathered in the square carrying books, tools, letters, and baskets of food. "
        "In another season a small laboratory recorded observations of light, pressure, and temperature, comparing measurements from different instruments. "
        "A traveler entering a new city learns by watching signs, listening to conversation, and remembering which streets lead toward familiar places.\n"
    ) * 45
    (eval_dir / "heldout.txt").write_text(corpus, encoding="utf-8")
    cfg.runtime.device = 'cpu'
    cfg.runtime.teacher_epochs = 1
    cfg.runtime.teacher_steps_per_sequence = 1
    cfg.runtime.train_steps_per_observation = 1
    cfg.runtime.observation_train_epochs = 1
    cfg.runtime.language_eval_every = 9999
    cfg.runtime.chat_train_steps = 0
    cfg.model.dim = 32
    cfg.model.layers = 1
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 512
    cfg.model.dropout = 0.0
    return cfg


def test_sharding_preserves_line_oriented_dialogue_structure():
    from aletheia.curriculum import split_lesson_text
    text = (
        "*CHI: more.\n"
        "%mor: adv.\n"
        "*MOT: more what?\n"
        "%int: question.\n"
        "*FAT: do you want juice?\n"
        "%add: child.\n"
        "*CHI: yes.\n"
    ) * 20
    shards = split_lesson_text(text, 160, title="childes", source="test://childes")
    assert len(shards) > 1
    assert any("*CHI: more.\n%mor: adv.\n*MOT: more what?" in shard.text for shard in shards)
    assert all("\n\n" not in shard.text for shard in shards)


def test_teacher_uses_large_source_and_holdout(tmp_path):
    cfg = small_cfg(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    text = ('Learning happens when a system predicts patterns in experience. ' * 200)
    result = agent.teach_input('text:' + text)[0]
    assert result['chars'] == len(text.strip())
    assert result['chunks'] > 20
    assert result['holdout_chunks'] > 0
    assert result['train_chunks'] > result['holdout_chunks']
    assert result['fixed_eval_bpb'] is not None and result['fixed_eval_bpb'] >= 0
    agent.close()


def test_language_readiness_does_not_use_training_loss(tmp_path):
    cfg = small_cfg(tmp_path)
    cfg.runtime.language_ready_min_tokens = 1
    cfg.runtime.language_ready_min_lessons = 1
    cfg.runtime.language_ready_min_bpb_gain = 999.0
    agent = AletheiaAgent(tmp_path, cfg)
    agent.last_language_loss = 0.0
    assert not agent._language_ready()
    agent.close()


def test_chat_question_retrieval_is_not_broken_by_punctuation(tmp_path):
    cfg = small_cfg(tmp_path)
    cfg.runtime.language_ready_min_tokens = 999999999
    agent = AletheiaAgent(tmp_path, cfg)
    agent.memory.add(kind='lesson', source='lesson', title='Water cycle', url='', text='Water moves through evaporation, condensation, precipitation, and collection.')
    assert agent._retrieve_experience('What does evaporation do?')
    assert agent._retrieve_experience('Hello') == []
    agent.close()


def test_long_context_trains_positions_beyond_256(tmp_path):
    cfg = small_cfg(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    chunks = agent._training_chunks('x' * 5000, max_chunks=0)
    assert any(chunk.shape[1] > 256 for chunk in chunks)
    assert max(chunk.shape[1] for chunk in chunks) <= 512
    agent.close()


def test_folder_traversal_handles_parent_paths_and_hidden_children(tmp_path):
    source = tmp_path / 'materials'
    (source / 'visible').mkdir(parents=True)
    (source / '.hidden').mkdir()
    (source / 'visible' / 'guide.txt').write_text('visible lesson', encoding='utf-8')
    (source / '.hidden' / 'secret.txt').write_text('hidden lesson', encoding='utf-8')
    lessons = list(iter_text_files(tmp_path / 'materials', max_chars=0))
    assert len(lessons) == 1
    assert lessons[0].text == 'visible lesson'


def test_localhost_and_private_hosts_are_rejected():
    assert WebExplorer.normalize_url('http://localhost:8000/') is None
    assert WebExplorer.normalize_url('http://127.0.0.1:8000/') is None
    assert WebExplorer.normalize_url('http://192.168.1.2/') is None


def test_fixed_eval_baseline_is_finite(tmp_path):
    cfg = small_cfg(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    metrics = agent.evaluate_fixed_reference()
    assert metrics['available'] and metrics['model_bpb'] > 0
    assert metrics['baseline_bpb'] > 0
    assert metrics['gain'] == metrics['baseline_bpb'] - metrics['model_bpb']
    assert len(metrics['model_state_sha256']) == 64
    assert metrics['tokenizer_sha256']
    assert len(metrics['config_sha256']) == 64
    assert metrics['evaluation_split'] == 'fixed-heldout-contiguous-tail'
    agent.close()


def test_checkpoint_source_uses_weights_only():
    source = Path('aletheia/model.py').read_text(encoding='utf-8')
    assert 'weights_only=True' in source
    assert 'weights_only=False' not in source


def test_policy_update_changes_action_head_on_positive_progress(tmp_path):
    cfg = small_cfg(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        before = [p.detach().clone() for p in agent.model.action_head.parameters()]
        state = agent.state.detach().clone()
        agent._learn_policy(state, 0, 1.0, True, "policy")
        after = [p.detach().clone() for p in agent.model.action_head.parameters()]
        assert any(not torch.equal(a, b) for a, b in zip(before, after))
    finally:
        agent.close()


def test_utf8_chunking_preserves_all_source_bytes(tmp_path):
    cfg = small_cfg(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    text = 'é漢字🙂' * 400
    chunks = agent._encode_chunks(text, max_chunks=0, context=512)
    # FINAL27 only appends EOS at the true document boundary, not every
    # context window. Count represented source bytes, not two assumed boundary
    # tokens per chunk.
    recovered = sum(
        sum(agent.tokenizer.token_byte_lengths(chunk[0].tolist()))
        for chunk in chunks
    )
    assert recovered == len(text.encode('utf-8'))
    agent.close()


def test_fixed_reference_uses_strong_baseline_and_has_real_holdout(tmp_path):
    from aletheia.evaluation import best_byte_baseline_bpb, byte_unigram_bpb, load_reference
    evaluation_dir = tmp_path / 'evaluation'
    evaluation_dir.mkdir()
    prose = (
        'The traveler crossed a quiet valley before sunrise, recording changes in weather and light. '
        'Later, the laboratory compared measurements from several instruments and discussed uncertainty in the results. '
        'A letter described a crowded market, a delayed train, and a conversation between strangers who exchanged directions. '
        'Another chapter followed a scientist explaining a difficult observation to a student who asked for clarification.\n'
    )
    (evaluation_dir / 'heldout.txt').write_text(prose * 250, encoding='utf-8')
    text = load_reference(tmp_path, 'evaluation/heldout.txt', 0)
    assert len(text.encode('utf-8')) >= 20000
    best = best_byte_baseline_bpb(text)
    unigram = byte_unigram_bpb(text)
    assert best['bpb'] <= unigram
    assert best['order'] in {1, 2, 3, 4}


def test_language_readiness_requires_margin_and_reference_size(tmp_path):
    cfg = small_cfg(tmp_path)
    cfg.runtime.language_ready_min_tokens = 1
    cfg.runtime.language_ready_min_lessons = 1
    cfg.runtime.language_ready_min_bpb_gain = 0.01
    cfg.runtime.language_ready_max_baseline_ratio = 0.90
    cfg.runtime.language_ready_min_eval_bytes = 50000
    agent = AletheiaAgent(tmp_path, cfg)
    agent.total_tokens = 100000
    agent.eval_reference_bytes = 50000
    agent.memory.add(kind='lesson', source='test', title='lesson', url='', text='This is a lesson about language and communication.')
    agent.last_eval_bpb = 3.65
    agent.eval_baseline_bpb = 3.72
    assert not agent._language_ready()
    agent.last_eval_bpb = 3.20
    assert agent._language_ready()
    agent.close()


def test_negative_progress_is_a_valid_policy_outcome(tmp_path):
    cfg = small_cfg(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        mask = torch.ones(len(agent.registry.items()), dtype=torch.bool)
        state = agent.state.detach().clone()
        before = [p.detach().clone() for p in agent.model.action_head.parameters()]
        agent._learn_policy(state, 0, -1.0, True, 'policy', mask)
        after = [p.detach().clone() for p in agent.model.action_head.parameters()]
        assert any(not torch.equal(a, b) for a, b in zip(before, after))
    finally:
        agent.close()


def test_zero_progress_is_a_valid_policy_outcome(tmp_path):
    cfg = small_cfg(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        mask = torch.ones(len(agent.registry.items()), dtype=torch.bool)
        state = agent.state.detach().clone()
        before = [p.detach().clone() for p in agent.model.action_head.parameters()]
        agent._learn_policy(state, 0, 0.0, True, 'policy', mask)
        after = [p.detach().clone() for p in agent.model.action_head.parameters()]
        assert any(not torch.equal(a, b) for a, b in zip(before, after))
    finally:
        agent.close()


def test_retrieval_ignores_generic_conversation_words(tmp_path):
    cfg = small_cfg(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        agent.memory.add(kind='external', source='p1', title='Plaque study', url='', text='A research article about artery plaque in young adults.')
        agent.memory.add(kind='external', source='p2', title='Match report', url='', text='The match ended after a close contest between two teams.')
        agent.memory.add(kind='external', source='p3', title='Water cycle', url='', text='Evaporation moves water into the atmosphere where condensation occurs.')
        assert agent._retrieve_experience('How are you?') == []
        matches = agent._retrieve_experience('What happens during evaporation?')
        assert matches and matches[0]['title'] == 'Water cycle'
    finally:
        agent.close()


def test_retrieval_query_uses_only_informative_terms(tmp_path):
    cfg = small_cfg(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        agent.memory.add(kind='external', source='p1', title='Water cycle', url='', text='Water evaporates, condenses, and returns through precipitation.')
        agent.memory.add(kind='external', source='p2', title='Match report', url='', text='The match ended after a close contest.')
        assert agent._retrieve_experience('How are you?') == []
        matches = agent._retrieve_experience('Can you explain water evaporation?')
        assert matches and matches[0]['title'] == 'Water cycle'
    finally:
        agent.close()


def test_redirect_to_private_target_is_blocked_before_second_request(monkeypatch):
    from requests import Response
    web = WebExplorer(WebConfig([], [], ['http', 'https']), 1, 1000, 'test', 0, 0)
    calls=[]
    monkeypatch.setattr(web._dns_adapter, '_resolve_public', lambda host: '93.184.216.34')
    monkeypatch.setattr(web, '_host_is_public', lambda host: host == 'public.example')
    def fake_get(url, **kwargs):
        calls.append(url)
        response=Response(); response.status_code=302; response.url=url; response.headers['Location']='http://localhost:1234/internal'
        return response
    monkeypatch.setattr(web.session, 'get', fake_get)
    try:
        import pytest
        with pytest.raises(ValueError):
            web._get('https://public.example/start')
        assert calls == ['https://public.example/start']
    finally:
        web.close()


def test_replay_excludes_holdout_chunks(tmp_path):
    cfg = small_cfg(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        text = 'abcdef ' * 500
        row_id = agent.memory.add(kind='external', source='test', title='long', url='', text=text, action_name='discover')
        rows = agent.memory.recent(1, kinds=('external',))
        tokens, actions, errors = agent._trajectory_from_rows(rows)
        all_chunks = agent._training_chunks(text, row_id, max_chunks=0)
        train_chunks, holdout = agent._split_train_holdout(all_chunks, cfg.runtime.teacher_holdout_fraction)
        assert len(tokens) == len(train_chunks)
        assert len(holdout) > 0
    finally:
        agent.close()


def test_fixed_evaluation_reference_is_not_taught_from_project_root(tmp_path):
    cfg = small_cfg(tmp_path)
    evaluation_dir = tmp_path / 'evaluation'
    evaluation_dir.mkdir(exist_ok=True)
    evaluation_file = evaluation_dir / 'heldout.txt'
    evaluation_file.write_text(('Evaluation prose for testing. ' * 2500), encoding='utf-8')
    (tmp_path / 'guide.txt').write_text('A separate lesson that should be taught.', encoding='utf-8')
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        results = agent.teach_file(tmp_path)
        assert all(r['source'] != str(evaluation_file.resolve()) for r in results)
        assert any(r['source'].endswith('guide.txt') for r in results)
    finally:
        agent.close()



def test_stronger_ngram_baseline_is_available(tmp_path):
    text = ("A varied paragraph about language, science, travel, and conversation. "
            "Another paragraph describes a laboratory notebook and a careful experiment. "
            "A third paragraph follows a traveler through streets, markets, and old buildings.\n" * 800)
    trigram, _ = tuned_byte_ngram_bpb(text, 3)
    fourgram, _ = tuned_byte_ngram_bpb(text, 4)
    best = best_byte_baseline_bpb(text)
    assert best['kind'] in {'unigram', 'bigram', 'trigram', 'fourgram'}
    assert best['bpb'] <= max(trigram, fourgram)


def test_one_lesson_does_not_make_generic_chat_terms_retrievable(tmp_path):
    cfg = small_cfg(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        agent.memory.add(kind='lesson', source='lesson', title='Communication guide', url='',
                         text='A communication guide explains turn taking, questions, answers, context, and clarification.')
        assert agent._retrieve_experience('How are you?') == []
        assert agent._retrieve_experience('What did you learn?') == []
        assert agent._retrieve_experience('What is turn taking?')
    finally:
        agent.close()


def test_policy_can_learn_zero_transfer_progress(tmp_path):
    cfg = small_cfg(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        mask = torch.ones(len(agent.registry.items()), dtype=torch.bool)
        state = agent.state.detach().clone()
        before = [p.detach().clone() for p in agent.model.action_head.parameters()]
        agent._learn_policy(state, 0, 0.0, True, 'policy', mask)
        after = [p.detach().clone() for p in agent.model.action_head.parameters()]
        assert any(not torch.equal(a, b) for a, b in zip(before, after))
    finally:
        agent.close()


def test_memory_rows_record_holdout_context(tmp_path):
    cfg = small_cfg(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        row_id = agent.memory.add(kind='external', source='test', title='probe', url='', text='context text ' * 400,
                                  train_context=128, holdout_indices='[0, 3]')
        row = agent.memory.recent(1)[0]
        assert row['train_context'] == 128
        assert row['holdout_indices'] == '[0, 3]'
        assert row['id'] == row_id
    finally:
        agent.close()


def test_missing_eval_corpus_does_not_enable_language(tmp_path):
    cfg = load_config(Path('config.json'))
    cfg.runtime.device = 'cpu'
    # This test needs a genuinely missing corpus. Do not use small_cfg(),
    # because small_cfg() intentionally creates evaluation/heldout.txt.
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        metrics = agent.evaluate_fixed_reference()
        assert metrics['available'] is False
        assert not agent._language_ready()
    finally:
        agent.close()


def test_evaluation_builder_targets_real_varied_prose(monkeypatch, tmp_path):
    import aletheia.eval_corpus as ec
    prose = (
        "At dawn the traveler crossed the valley and recorded the changing weather in a notebook."
        " Later, the laboratory compared several instruments and discussed uncertainty in the measurements."
        " A letter described a crowded market, a delayed train, and a conversation between strangers."
        " Another chapter followed a student asking a teacher for clarification about an unfamiliar idea."
    )
    monkeypatch.setattr(ec, '_fetch', lambda url: prose * 5000)
    result = ec.build_reference(tmp_path / 'evaluation' / 'heldout.txt', target_bytes=80000, sources=[('one','https://a'),('two','https://b'),('three','https://c'),('four','https://d')])
    assert 50000 <= result['bytes'] <= 80000
    assert len(result['sources']) == 4
    assert (tmp_path / 'evaluation' / 'heldout.sources.json').exists()


def test_evaluation_builder_fills_target_with_variable_paragraphs(monkeypatch, tmp_path):
    import aletheia.eval_corpus as ec
    paras = []
    for i in range(240):
        paras.append((
            f"Paragraph {i} describes a different scene with weather, science, conversation, travel, "
            f"measurement, letters, gardens, rivers, and changing plans. " * (1 + (i % 5))
        ).strip())
    prose = "\n\n".join(paras)
    monkeypatch.setattr(ec, '_fetch', lambda url: prose)
    result = ec.build_reference(tmp_path / 'evaluation' / 'heldout.txt', target_bytes=80000, sources=[('one','https://a'),('two','https://b'),('three','https://c'),('four','https://d')])
    assert result['bytes'] >= 50000
    assert result['bytes'] <= 80000
    assert result['sources']
    assert (tmp_path / 'evaluation' / 'heldout.sources.json').exists()



def test_evaluation_tail_is_mixed_across_sources(monkeypatch, tmp_path):
    import aletheia.eval_corpus as ec
    sources = [(f"source-{i}", f"https://example.org/{i}") for i in range(4)]
    payloads = {
        url: (f"SOURCE_{i} " + ("Distinct prose about science, travel, communication, and history, with a different scene and setting described in careful detail. " * 4) + "\n\n") * 500
        for i, (_, url) in enumerate(sources)
    }
    monkeypatch.setattr(ec, '_fetch', lambda url: payloads[url])
    result = ec.build_reference(tmp_path / 'evaluation' / 'heldout.txt', target_bytes=50000, sources=sources)
    text = (tmp_path / 'evaluation' / 'heldout.txt').read_text(encoding='utf-8')
    tail = text[int(len(text) * 0.8):]
    markers = {f"SOURCE_{i}" for i in range(4) if f"SOURCE_{i}" in tail}
    assert len(markers) >= 3
    assert result['bytes'] == 50000


def test_evaluation_paragraph_hashes_protect_teaching(tmp_path):
    import json
    from aletheia.eval_corpus import build_reference
    import aletheia.eval_corpus as ec
    source = "Evaluation paragraph alpha. " + "x " * 120 + "\n\nAnother distinct paragraph. " + "y " * 120
    original = ec._fetch
    ec._fetch = lambda url: source * 100
    try:
        target = build_reference(
            tmp_path / 'evaluation' / 'heldout.txt',
            target_bytes=50000,
            sources=[('source-a', 'https://example.org/source-a'), ('source-b', 'https://example.org/source-b')],
        )
    finally:
        ec._fetch = original
    cfg = load_config(Path('config.json'))
    cfg.runtime.device = 'cpu'
    from aletheia.agent import AletheiaAgent
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        lesson = agent._exclude_evaluation_text(source)
        assert 'Evaluation paragraph alpha.' not in lesson
        assert 'Another distinct paragraph.' not in lesson
    finally:
        agent.close()

def test_evaluation_builder_does_not_pad_failed_sources(monkeypatch, tmp_path):
    import aletheia.eval_corpus as ec
    payloads = {
        f"https://example.org/{i}": (
            "A long varied paragraph about science, communication, history, travel, and measurement. " * 50
            + "\n\n"
        ) * 40
        for i in range(8)
    }
    sources = [(f"source-{i}", f"https://example.org/{i}") for i in range(10)]

    def fake_fetch(url):
        if url.endswith("/8") or url.endswith("/9"):
            raise OSError("simulated download failure")
        return payloads[url]

    monkeypatch.setattr(ec, "_fetch", fake_fetch)
    result = ec.build_reference(
        tmp_path / "evaluation" / "heldout.txt",
        target_bytes=80000,
        sources=sources,
    )
    raw = (tmp_path / "evaluation" / "heldout.txt").read_bytes()
    assert result["errors"] and len(result["errors"]) == 2
    assert 50000 <= len(raw) <= 80000
    assert not raw.endswith(b"\n" * 100)


def test_evaluation_builder_fails_when_too_many_sources_fail(monkeypatch, tmp_path):
    import aletheia.eval_corpus as ec
    prose = (
        "A long varied paragraph about science, communication, history, travel, and measurement. " * 40
        + "\n\n"
    ) * 100
    sources = [(f"source-{i}", f"https://example.org/{i}") for i in range(10)]

    def fake_fetch(url):
        if url.endswith(("/7", "/8", "/9")):
            raise OSError("simulated download failure")
        return prose

    monkeypatch.setattr(ec, "_fetch", fake_fetch)
    import pytest
    with pytest.raises(RuntimeError, match="loaded only 7/10 sources"):
        ec.build_reference(
            tmp_path / "evaluation" / "heldout.txt",
            target_bytes=80000,
            sources=sources,
        )


def test_evaluation_overlap_ignores_whitespace_and_truncated_tail(tmp_path):
    full = (
        "Evaluation paragraph alpha contains enough distinctive wording about science, "
        "communication, measurement, weather, and travel to make accidental overlap unlikely. "
        "The paragraph continues with a second sentence that is intentionally longer than the held-out slice."
    )
    truncated = full[:180]
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    evaluation_file = evaluation_dir / "heldout.txt"
    evaluation_file.write_text(
        truncated + "\n\n" + ("Independent reference prose. " * 2500),
        encoding="utf-8",
    )

    cfg = load_config(Path("config.json"))
    cfg.runtime.device = "cpu"
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        wrapped = full.replace("science, communication", "science,\ncommunication")
        filtered = agent._exclude_evaluation_text(wrapped)
        assert "Evaluation paragraph alpha" not in filtered
    finally:
        agent.close()

def test_autonomous_web_ingest_uses_evaluation_contamination_filter(tmp_path):
    cfg = load_config(Path("config.json"))
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    evaluation_file = evaluation_dir / "heldout.txt"
    contaminated = (
        "This is a distinctive evaluation paragraph about science, measurement, and communication "
        "that should never become part of the teaching memory.\n\n"
        "A safe paragraph from a genuinely external page that should remain teachable."
    )
    evaluation_file.write_text(contaminated.split("\n\n")[0] + "\n\n", encoding="utf-8")
    cfg.runtime.device = "cpu"
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        from aletheia.web import Page
        page = Page(
            url="https://example.org/page",
            title="External page",
            text=contaminated,
            status_code=200,
            links=[],
        )
        result = agent._ingest(page, source="search", action_name="search")
        assert result["learned"]
        rows = agent.memory.recent(5, kinds=("external",))
        assert rows
        latest = rows[-1]
        assert "Distinctive evaluation paragraph" not in latest["text"]
        assert "safe paragraph" in latest["text"]
    finally:
        agent.close()



def test_chunk_memory_retrieves_the_relevant_part(tmp_path):
    cfg = small_cfg(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        memory_id = agent.memory.add(
            kind='lesson', source='lesson', title='Long lesson', url='',
            text='First section about rivers and weather.\n\nSecond section about orbital mechanics and satellites.',
        )
        agent.memory.replace_chunks(memory_id, [
            'First section about rivers and weather.',
            'Second section about orbital mechanics and satellites.',
        ])
        chunks = agent.memory.search_chunks('satellites orbital', 4)
        assert chunks
        assert any('satellites' in row['text'] for row in chunks)
    finally:
        agent.close()


def test_dialogue_extraction_builds_structural_response_pairs():
    from aletheia.dialogue import dialogue_response_pairs, dialogue_training_sequences, extract_dialogue_turns
    text = 'Person A: Hello there friend.\nPerson B: Hello to you too.\nPerson A: How are you today?\nPerson B: I am doing quite well.'
    runs = extract_dialogue_turns(text)
    pairs = dialogue_response_pairs(text)
    examples = dialogue_training_sequences(text)
    assert runs and pairs and examples
    assert pairs[0][1] == 'Hello to you too.'
    assert 'Turn B:' in examples[0]
    assert len(runs) == 1
    examples = dialogue_training_sequences(text, max_examples=8)
    assert len(examples) >= 2
    assert any('Turn A:' in item and 'Turn B:' in item for item in examples)
    assert any('How are you today?' in item and 'I am doing quite well.' in item for item in examples)





def test_dialogue_parser_ignores_transcript_dependent_tiers():
    from aletheia.dialogue import extract_dialogue_turns, dialogue_response_pairs
    text = (
        "*CHI: hi.\n"
        "%int: greeting.\n"
        "*MOT: hello.\n"
        "%mor: intj.\n"
        "*CHI: more juice.\n"
        "*MOT: more what?\n"
        "%add: object.\n"
        "*CHI: apple juice.\n"
        "*MOT: here you go.\n"
    )
    runs = extract_dialogue_turns(text, min_turn_chars=1)
    pairs = dialogue_response_pairs(text, max_examples=20)
    assert runs
    assert pairs
    assert all(turn.speaker in {"*CHI", "*MOT"} for run in runs for turn in run)
    assert all("%int" not in prompt and "%mor" not in prompt and "%add" not in prompt for prompt, _ in pairs)


def test_childes_annotations_do_not_break_multi_party_dialogue_runs():
    from aletheia.dialogue import dialogue_response_pairs, extract_dialogue_turns
    text = (
        "@Begin\n"
        "@Participants: *CHI Child, *MOT Mother, *FAT Father\n"
        "*CHI: more.\n"
        "%mor: adv.\n"
        "*MOT: more what?\n"
        "%int: question.\n"
        "*FAT: do you want juice?\n"
        "%add: child.\n"
        "*CHI: yes.\n"
        "@End\n"
        "@Begin\n"
        "*CHI: another turn.\n"
        "*MOT: another reply.\n"
        "*CHI: one more turn.\n"
        "*MOT: one more reply.\n"
        "@End\n"
    )
    runs = extract_dialogue_turns(text, min_turn_chars=1)
    pairs = dialogue_response_pairs(text, max_examples=50, seed=7)
    assert len(runs) == 2
    assert len(runs[0]) == 4
    assert len(runs[1]) == 4
    assert len(pairs) >= 5
    assert all(not any(tier in prompt for tier in ("%mor", "%int", "%add")) for prompt, _ in pairs)


def test_multi_party_dialogue_normalizes_current_response_to_turn_b():
    from aletheia.dialogue import dialogue_training_sequences
    text = (
        "Child: more.\n"
        "Mother: more what?\n"
        "Father: do you want juice?\n"
        "Child: yes.\n"
    )
    examples = dialogue_training_sequences(text, max_examples=20)
    assert len(examples) == 3
    assert all("\nTurn B:\n" in example for example in examples)
    assert any("Turn C:\nmore what?" in example for example in examples)


def test_dialogue_teaching_extracts_childes_style_turns(tmp_path):
    cfg = small_cfg(tmp_path)
    cfg.runtime.teacher_max_chunks = 2
    cfg.runtime.teacher_shard_max_chars = 220
    cfg.runtime.dialogue_max_examples = 8
    cfg.runtime.dialogue_passes = 1
    cfg.runtime.dialogue_train_epochs = 1
    cfg.runtime.conversation_replay_examples = 0
    cfg.runtime.dialogue_replay_multiplier = 0.0
    cfg.runtime.dialogue_alignment_weight = 0.0
    source = tmp_path / "childes.txt"
    block = (
        "@Begin\n"
        "@Participants: *CHI Child, *MOT Mother, *FAT Father\n"
        "*CHI: more.\n"
        "%mor: adv.\n"
        "*MOT: more what?\n"
        "%int: question.\n"
        "*FAT: do you want juice?\n"
        "*CHI: yes.\n"
        "@End\n"
    )
    source.write_text(block * 12, encoding="utf-8")
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        results = agent.teach_file(source, dialogue=True, epochs=1)
        assert len(results) > 1
        assert all(result["dialogue_examples"] > 0 for result in results)
        assert agent.dialogue_examples_seen > 0
    finally:
        agent.close()


def test_replay_builder_honors_target_and_covers_multiple_prior_lessons(tmp_path):
    cfg = small_cfg(tmp_path)
    cfg.runtime.teacher_sequence_length = 2
    cfg.runtime.replay_max_sequences_per_lesson = 2
    cfg.runtime.replay_lesson_pool = 0
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        ids = []
        for index in range(5):
            text = (
                f"Lesson {index} describes observations, explanations, experiments, context, "
                "memory, language, and changes over time. " * 35
            )
            ids.append(
                agent.memory.add(
                    kind="lesson",
                    source=f"lesson-{index}",
                    title=f"Lesson {index}",
                    url="",
                    text=text,
                    train_context=cfg.model.context,
                )
            )

        replay, source_ids = agent._build_replay_sequences(
            6,
            exclude_ids={ids[-1]},
        )
        assert len(replay) == 6
        assert ids[-1] not in source_ids
        assert len(source_ids) >= 3

    finally:
        agent.close()


def test_replay_minimum_is_applied_and_reported(tmp_path):
    cfg = small_cfg(tmp_path)
    cfg.runtime.replay_mix_fraction = 0.01
    cfg.runtime.replay_min_sequences = 3
    cfg.runtime.teacher_sequence_length = 2
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        for index in range(4):
            text = (
                f"Prior lesson {index} contains enough material for replay and "
                "should remain available during future teaching. " * 30
            )
            agent.memory.add(
                kind="lesson",
                source=f"lesson-{index}",
                title=f"Prior lesson {index}",
                url="",
                text=text,
                train_context=cfg.model.context,
            )

        current = agent._encode_chunks(
            "A fresh lesson introduces new material that should be learned while older lessons are replayed. " * 30,
            max_chunks=4,
            context=cfg.model.context,
        )
        sequences = agent._as_sequences(current, 2)
        calls = []

        def fake_batch(batch):
            calls.append(len(batch))
            return 1.0

        agent._batch_trajectory_sequences = fake_batch
        agent._train_sequence_collection(
            sequences,
            epochs=1,
            replay=True,
            minimum_replay_sequences=3,
        )
        stats = agent.last_replay_stats
        assert stats["used"] >= 3
        assert stats["per_epoch"] == [stats["used"]]
        assert calls == [len(sequences) + stats["used"]]
    finally:
        agent.close()


def test_training_token_accounting_separates_unique_and_processed_tokens(tmp_path):
    cfg = small_cfg(tmp_path)
    cfg.runtime.teacher_epochs = 2
    cfg.runtime.teacher_steps_per_sequence = 1
    cfg.runtime.replay_mix_fraction = 0.0
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        text = 'Learning systems predict the next observation. ' * 80
        first = agent.teach_input('text:' + text)[0]
        unique_after_first = agent.total_tokens
        processed_after_first = agent.processed_tokens
        second = agent.teach_input('text:' + text)[0]
        assert first['unique_tokens_added'] > 0
        assert first['unique_tokens_added'] < len(text.encode('utf-8'))
        assert second['unique_tokens_added'] == 0
        assert agent.total_tokens == unique_after_first
        assert agent.processed_tokens > processed_after_first
        assert agent.training_steps > 0
    finally:
        agent.close()


def test_generation_updates_recurrent_state(tmp_path):
    cfg = small_cfg(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        import torch
        state = agent.state.detach().clone()
        _, new_state, metrics = agent.model.generate_with_state(
            'Turn A:\nHello\nTurn B:\n',
            max_new_tokens=2,
            temperature=0.9,
            top_k=8,
            device=agent.device,
            state=state,
        )
        assert not torch.equal(state, new_state)
        assert metrics['generated_tokens'] >= 1
    finally:
        agent.close()


def test_generation_candidate_reranking_uses_only_local_model(tmp_path):
    cfg = small_cfg(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        text, state, metrics = agent.model.generate_candidates(
            'Turn A:\nHello\nTurn B:\n',
            candidate_count=2,
            max_new_tokens=2,
            temperature=0.9,
            top_k=8,
            device=agent.device,
            state=agent.state,
        )
        assert isinstance(text, str)
        assert tuple(state.shape) == tuple(agent.state.shape)
        assert metrics['candidate_count'] == 2
        assert len(metrics['candidate_bpb']) == 2
    finally:
        agent.close()


def test_chat_does_not_dump_single_word_memory_quotes(tmp_path):
    cfg = small_cfg(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        agent.memory.add(
            kind='lesson', source='lesson', title='Greeting transcript', url='',
            text='Person A: Hello.\nPerson B: Hello.',
        )
        reply = agent.communicate('Hello')
        assert '[quoted]' not in reply
        assert 'Greeting transcript' not in reply
    finally:
        agent.close()


def test_chat_can_generate_experimental_replies_before_language_readiness(tmp_path, monkeypatch):
    cfg = small_cfg(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        called = {}
        def experimental(*args, **kwargs):
            called.update(kwargs)
            return ('A tentative response about communication from my current learned state.', agent.state.detach().clone(), {'candidate_count': kwargs['candidate_count'], 'generated_tokens': 12})
        monkeypatch.setattr(agent.model, 'generate_candidates', experimental)
        agent.total_tokens = 10_000_000
        agent.memory.add(kind='lesson', source='lesson', title='lesson one', url='', text='A long lesson about language and communication that is useful evidence.')
        agent.memory.add(kind='lesson', source='lesson2', title='lesson two', url='', text='A second lesson about language and communication and context.')
        agent.processed_tokens = 10_000_000
        agent.last_eval_bpb = 4.0
        agent.eval_baseline_bpb = 3.0
        agent.eval_reference_bytes = 50_000
        assert not agent._language_ready()
        reply = agent.communicate('What did you learn about communication context?', max_tokens=200)
        assert reply.startswith('[experimental] ')
        assert called['candidate_count'] == 1
        assert called['max_new_tokens'] <= cfg.runtime.chat_unready_max_response_tokens
        assert not agent._language_ready()
    finally:
        agent.close()


def test_early_chat_generation_can_be_disabled_explicitly(tmp_path, monkeypatch):
    cfg = small_cfg(tmp_path)
    cfg.runtime.chat_allow_unready_generation = False
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        called = {'value': False}
        def forbidden(*args, **kwargs):
            called['value'] = True
            raise AssertionError('generation disabled by configuration')
        monkeypatch.setattr(agent.model, 'generate_candidates', forbidden)
        reply = agent.communicate('An experimental message with no stored answer.')
        assert '[experimental]' not in reply
        assert not called['value']
    finally:
        agent.close()


def test_evidence_and_generated_replies_keep_explicit_labels(tmp_path, monkeypatch):
    cfg = small_cfg(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        agent.memory.add(kind='lesson', source='lesson', title='Water cycle', url='', text='Evaporation moves water into the atmosphere before condensation returns it.')
        quoted = agent.communicate('What happens during evaporation?')
        assert quoted.startswith('[quoted] From Water cycle:')

        agent.last_eval_bpb = 2.5
        agent.eval_baseline_bpb = 3.5
        agent.eval_reference_bytes = 50_000
        agent.total_tokens = 2_000_000
        agent.config.runtime.language_ready_min_lessons = 2
        agent.memory.add(kind='lesson', source='lesson2', title='Science', url='', text='Science describes observations and measurements carefully over time.')
        monkeypatch.setattr(agent.model, 'generate_candidates', lambda *args, **kwargs: ('A useful generated answer.', agent.state.detach().clone(), {'candidate_count': 1, 'generated_tokens': 20}))
        generated = agent.communicate('What is science?')
        assert generated.startswith('[generated] ')
    finally:
        agent.close()


def test_single_long_term_can_retrieve_but_short_greeting_cannot(tmp_path):
    cfg = small_cfg(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        agent.memory.add(kind='lesson', source='lesson', title='Photosynthesis', url='', text='Photosynthesis converts light energy into chemical energy in plants.')
        assert agent._retrieve_experience('photosynthesis')
        assert agent._retrieve_experience('Hello') == []
    finally:
        agent.close()


def test_dialogue_parsing_is_opt_in_for_teaching(tmp_path):
    cfg = small_cfg(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        text='Person A: Hello there friend.\nPerson B: Hello to you too.\nPerson A: How are you today?\nPerson B: I am doing quite well.'
        normal = agent.teach_input('text:' + text)[0]
        assert normal['dialogue_examples'] == 0
        assert agent.dialogue_examples_seen == 0
        dialogue = agent.teach_input('text:' + text, dialogue=True)[0]
        assert dialogue['dialogue_examples'] >= 2
    finally:
        agent.close()



def test_dialogue_parser_requires_repeated_labels_for_two_turn_runs():
    from aletheia.dialogue import dialogue_training_sequences, extract_dialogue_turns
    fake = (
        "Section 1: Communication and context are important in ordinary prose.\n"
        "Example B: A second sentence with a colon that looks like a speaker label.\n"
    )
    assert extract_dialogue_turns(fake) == []
    assert dialogue_training_sequences(fake) == []


def test_dialogue_parser_accepts_repeated_two_turn_pairs():
    from aletheia.dialogue import extract_dialogue_turns
    text = (
        "Question: What happens during rain?\n"
        "Answer: Water falls from clouds as precipitation.\n"
        "Question: What happens after rain?\n"
        "Answer: Water collects, flows, or enters the soil.\n"
    )
    runs = extract_dialogue_turns(text)
    assert len(runs) == 1
    assert len(runs[0]) == 4


def test_whitespace_only_reteach_does_not_add_unique_training_tokens(tmp_path):
    cfg = small_cfg(tmp_path)
    cfg.runtime.teacher_epochs = 1
    cfg.runtime.teacher_steps_per_sequence = 1
    cfg.runtime.replay_mix_fraction = 0.0
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        text = 'Learning systems predict patterns from experience. ' * 20
        first = agent.teach_input('text:' + text)[0]
        unique_after_first = agent.total_tokens
        second = agent.teach_input('text:' + text.replace('experience. ', 'experience.  '))[0]
        assert first['unique_tokens_added'] > 0
        assert second['unique_tokens_added'] == 0
        assert agent.total_tokens == unique_after_first
    finally:
        agent.close()

def test_dialogue_parser_rejects_short_fake_colon_turns():
    from aletheia.dialogue import dialogue_training_sequences, extract_dialogue_turns
    fake='Turn A: yes, of\nTurn B: no.'
    # Direct extraction keeps its conservative default; explicit dialogue training
    # can opt into short turns only after structural validation.
    assert extract_dialogue_turns(fake) == []
    assert dialogue_training_sequences(fake) == []


def test_dialogue_training_accepts_blank_lines_between_turns():
    from aletheia.dialogue import dialogue_training_sequences
    text = (
        'Human: Hi.\n\n'
        'Assistant: Hello!\n\n'
        'Human: Thanks.\n\n'
        'Assistant: You are welcome.\n'
    )
    examples = dialogue_training_sequences(text, max_examples=8)
    assert len(examples) >= 2


def test_dialogue_training_accepts_short_valid_turns():
    from aletheia.dialogue import dialogue_training_sequences
    text = (
        'Human: Hi.\n'
        'Assistant: Hello!\n'
        'Human: Thanks.\n'
        'Assistant: You are welcome.\n'
    )
    examples = dialogue_training_sequences(text, max_examples=8)
    assert len(examples) >= 2
    assert any('Hi.' in item and 'Hello!' in item for item in examples)
    assert all('Turn A:' in item and 'Turn B:' in item for item in examples)


def test_continuation_scoring_uses_sliding_context_windows(tmp_path):
    cfg=small_cfg(tmp_path)
    agent=AletheiaAgent(tmp_path, cfg)
    try:
        import torch
        def fake_encode(tokens, state=None):
            logits=torch.zeros(tokens.size(0), tokens.size(1), agent.model.cfg.vocab_size)
            logits[:,:,ord('a')]=10.0
            hidden=torch.zeros(tokens.size(0), tokens.size(1), agent.model.cfg.dim)
            return logits, hidden
        agent.model.encode_sequence=fake_encode
        prompt='P'*1400
        continuation='a'*64
        bpb=agent.model.continuation_bpb(prompt, continuation, agent.device)
        assert bpb < 1.0
    finally:
        agent.close()


def test_candidate_reranking_considers_repetition_penalty(tmp_path, monkeypatch):
    cfg=small_cfg(tmp_path)
    agent=AletheiaAgent(tmp_path, cfg)
    try:
        import torch
        outputs=iter([('aaaaaaaaaaaaaaaa', agent.state.detach().clone(), {'generated_tokens':16}), ('abcdefghijklmnop', agent.state.detach().clone(), {'generated_tokens':16})])
        monkeypatch.setattr(agent.model, 'generate_with_state', lambda *args, **kwargs: next(outputs))
        monkeypatch.setattr(agent.model, 'continuation_bpb', lambda *args, **kwargs: 1.0)
        text, _, metrics=agent.model.generate_candidates('hello '*300, candidate_count=2, max_new_tokens=16, temperature=0.8, top_k=8, device=agent.device, state=agent.state)
        assert text == 'abcdefghijklmnop'
        assert metrics['selected_repeat_trigram_rate'] == min(metrics['candidate_repeat_trigram_rate'])
        assert max(metrics['candidate_repeat_trigram_rate']) > metrics['selected_repeat_trigram_rate']
    finally:
        agent.close()


def test_gradient_checkpointing_really_executes_training_graph():
    from aletheia.config import ModelConfig
    from aletheia.model import AletheiaModel
    from aletheia.tokenizer import encode
    cfg=ModelConfig(dim=32,layers=2,heads=4,ffn=64,recurrent_dim=48,context=64,dropout=0.0,gradient_checkpointing=True)
    model=AletheiaModel(cfg,['teacher'])
    tokens=torch.tensor([encode('gradient checkpointing works')],dtype=torch.long)
    model.train()
    loss=model.language_loss(tokens)
    loss.backward()
    assert torch.isfinite(loss)

def test_replay_rechunks_old_memories_for_current_model_context(tmp_path):
    cfg = small_cfg(tmp_path)
    cfg.model.context = 256
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        row_id = agent.memory.add(
            kind='lesson', source='old', title='Old lesson', url='',
            text='contextual material ' * 300, train_context=512, holdout_indices='[0, 1, 2, 3]',
        )
        row = agent.memory.recent_lessons(1)[0]
        train, holdout = agent._chunks_for_memory_row(row)
        assert train and holdout
        assert max(chunk.size(1) for chunk in train + holdout) <= 256
        assert row_id == row['id']
    finally:
        agent.close()


def test_cuda_amp_dtype_prefers_bfloat16_when_supported(monkeypatch):
    import torch
    from aletheia.agent import AletheiaAgent

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True, raising=False)
    assert AletheiaAgent._select_amp_dtype(torch.device("cuda")) is torch.bfloat16


def test_cuda_amp_dtype_falls_back_to_float16(monkeypatch):
    import torch
    from aletheia.agent import AletheiaAgent

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False, raising=False)
    assert AletheiaAgent._select_amp_dtype(torch.device("cuda")) is torch.float16


def test_reloaded_agent_optimizer_updates_reloaded_model(tmp_path):
    from aletheia.agent import AletheiaAgent
    from aletheia.config import load_config

    cfg = load_config(Path("config.json"))
    cfg.runtime.device = "cpu"
    cfg.runtime.mixed_precision = False
    cfg.model.dim = 32
    cfg.model.layers = 2
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 64
    cfg.model.gradient_checkpointing = False

    first = AletheiaAgent(tmp_path, cfg)
    text = "Checkpoint reload must keep the optimizer attached to the current model. " * 8
    chunks = first._encode_chunks(text, max_chunks=2, context=cfg.model.context)
    first._batch_trajectory_sequences(first._as_sequences(chunks, 1))
    first.save()
    first.close()

    second = AletheiaAgent(tmp_path, cfg)
    try:
        second.model.train()
        before = {
            name: parameter.detach().clone()
            for name, parameter in second.model.named_parameters()
            if parameter.requires_grad
        }
        second._batch_trajectory_sequences(second._as_sequences(chunks, 1))
        deltas = [
            float((parameter.detach() - before[name]).abs().max().cpu())
            for name, parameter in second.model.named_parameters()
            if parameter.requires_grad
        ]
        assert max(deltas) > 0.0
    finally:
        second.close()


def test_dialogue_training_is_forward_only_and_contextual():
    from aletheia.dialogue import dialogue_training_sequences
    text = (
        'Human: Hi.\n'
        'Assistant: Hello!\n'
        'Human: How are you?\n'
        'Assistant: I am well.\n'
    )
    examples = dialogue_training_sequences(text, max_examples=8)
    assert len(examples) == 2
    assert 'Turn A:\nHi.\nTurn B:\nHello!' in examples[0]
    assert 'Turn A:\nHi.\nTurn B:\nHello!\nTurn A:\nHow are you?\nTurn B:\nI am well.' in examples[1]
    assert not any('Turn B:\nHello!\nTurn A:' in example and example.endswith('Hi.\n') for example in examples)


def test_dialogue_training_can_roll_back_on_fixed_eval_regression(tmp_path, monkeypatch):
    cfg = small_cfg(tmp_path)
    cfg.runtime.dialogue_train_epochs = 1
    cfg.runtime.dialogue_max_examples = 4
    cfg.runtime.dialogue_replay_multiplier = 1.0
    cfg.runtime.conversation_replay_examples = 1
    cfg.runtime.dialogue_learning_rate_scale = 0.25
    cfg.runtime.dialogue_max_fixed_regression = 0.01
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        text = ('Human: Hi.\nAssistant: Hello!\nHuman: Thanks.\nAssistant: You are welcome.\n') * 6
        before = {k: v.detach().clone() for k, v in agent.model.state_dict().items()}
        original_train = agent._train_sequence_collection
        train_calls = [0]
        def wrapped_train(*args, **kwargs):
            train_calls[0] += 1
            if train_calls[0] == 1:
                return 0.0
            return original_train(*args, **kwargs)
        monkeypatch.setattr(agent, '_train_sequence_collection', wrapped_train)
        calls = [1.0, 1.0, 0.0, 0.0]  # lesson baseline, dialogue baseline, dialogue after, final
        def fake_eval():
            value = calls.pop(0) if calls else 0.0
            return {'model_bpb': 4.0, 'baseline_bpb': 3.0, 'gain': value, 'reference_bytes': 80000}
        monkeypatch.setattr(agent, 'evaluate_fixed_reference', fake_eval)
        result = agent.teach_input('text:' + text, dialogue=True)[0]
        assert result['dialogue_examples'] > 0
        assert result['dialogue_accepted'] is False
        after = agent.model.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before)
    finally:
        agent.close()


def test_dialogue_regression_tolerance_allows_small_change(tmp_path, monkeypatch):
    cfg = small_cfg(tmp_path)
    cfg.runtime.dialogue_train_epochs = 1
    cfg.runtime.dialogue_max_examples = 4
    cfg.runtime.dialogue_replay_multiplier = 1.0
    cfg.runtime.conversation_replay_examples = 1
    cfg.runtime.dialogue_max_fixed_regression = 0.02
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        text = ('Human: Hi.\nAssistant: Hello!\nHuman: Thanks.\nAssistant: You are welcome.\n') * 6
        calls = [1.0, 1.0, 0.99, 0.99]
        def fake_eval():
            value = calls.pop(0) if calls else 0.99
            return {'model_bpb': 4.0, 'baseline_bpb': 3.0, 'gain': value, 'reference_bytes': 80000}
        monkeypatch.setattr(agent, 'evaluate_fixed_reference', fake_eval)
        result = agent.teach_input('text:' + text, dialogue=True)[0]
        assert result['dialogue_examples'] > 0
        assert result['dialogue_accepted'] is True
        assert result['dialogue_fixed_gain_delta'] == pytest.approx(-0.01)
    finally:
        agent.close()



def test_ordinary_lesson_rolls_back_on_large_fixed_eval_regression(tmp_path):
    cfg = small_cfg(tmp_path)
    cfg.runtime.lesson_rollback_on_fixed_regression = True
    cfg.runtime.lesson_max_fixed_regression = 0.05
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        text = "A careful learner compares observations, records evidence, and revises explanations when results change. " * 80
        original = agent.model.state_dict()["token_embedding.weight"].detach().clone()
        calls = iter([{"gain": 0.0}, {"gain": -0.20}, {"gain": 0.0}])
        agent.evaluate_fixed_reference = lambda: next(calls)

        def fake_train(sequences, **kwargs):
            with torch.no_grad():
                agent.model.token_embedding.weight.add_(0.5)
            return 7.5

        agent._train_sequence_collection = fake_train
        result = agent.teach_text(Lesson(title="rollback-test", source="rollback-test", text=text), epochs=1)
        restored = agent.model.state_dict()["token_embedding.weight"].detach().clone()
        assert result["lesson_accepted"] is False
        assert result["lesson_rollback_threshold"] == pytest.approx(0.05)
        assert result["lesson_fixed_gain_delta"] == pytest.approx(-0.20)
        assert torch.equal(original, restored)
        rows = agent.memory.recent(5, kinds=("lesson",))
        assert rows[-1]["action_name"] == "teacher_rollback"
    finally:
        agent.close()


def test_ordinary_lesson_can_be_accepted_when_fixed_eval_is_stable(tmp_path):
    cfg = small_cfg(tmp_path)
    cfg.runtime.lesson_rollback_on_fixed_regression = True
    cfg.runtime.lesson_max_fixed_regression = 0.05
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        text = "A learner observes patterns in a changing environment and updates an internal model from evidence. " * 80
        original = agent.model.state_dict()["token_embedding.weight"].detach().clone()
        calls = iter([{"gain": -0.10}, {"gain": -0.08}])
        agent.evaluate_fixed_reference = lambda: next(calls)

        def fake_train(sequences, **kwargs):
            with torch.no_grad():
                agent.model.token_embedding.weight.add_(0.01)
            return 2.0

        agent._train_sequence_collection = fake_train
        result = agent.teach_text(Lesson(title="accept-test", source="accept-test", text=text), epochs=1)
        changed = agent.model.state_dict()["token_embedding.weight"].detach().clone()
        assert result["lesson_accepted"] is True
        assert result["lesson_fixed_gain_delta"] == pytest.approx(0.02)
        assert not torch.equal(original, changed)
        rows = agent.memory.recent(5, kinds=("lesson",))
        assert rows[-1]["action_name"] == "teacher"
    finally:
        agent.close()


def test_reset_learning_exits_without_autonomous_run(monkeypatch, tmp_path):
    import sys
    import aletheia.cli as cli

    calls = []

    class FakeAgent:
        @staticmethod
        def reset_storage(root):
            calls.append(("reset", root))
            return None

        def __init__(self, *args, **kwargs):
            calls.append(("init", args, kwargs))

        def run(self, *args, **kwargs):
            calls.append(("run", args, kwargs))

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(cli, "AletheiaAgent", FakeAgent)
    monkeypatch.setattr(
        sys,
        "argv",
        ["aletheia", "--root", str(tmp_path), "--reset-learning"],
    )

    cli.main()

    assert calls == [("reset", tmp_path)]


def test_response_alignment_loss_backpropagates(tmp_path):
    cfg = small_cfg(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        pairs = [
            ('Turn A: The train arrived at the station.', 'Turn B: The passengers got off.'),
            ('Turn A: The kettle is boiling.', 'Turn B: Please turn off the heat.'),
        ]
        contexts = [agent._encode_text(item[0]) for item in pairs]
        responses = [agent._encode_text(item[1]) for item in pairs]
        agent.model.train()
        with agent._autocast():
            loss = agent.model.response_alignment_loss(contexts, responses, temperature=0.12)
        assert torch.isfinite(loss)
        agent.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert any(p.grad is not None for p in agent.model.parameters() if p.requires_grad)
    finally:
        agent.close()


def test_autonomous_web_learning_uses_replay_and_rolls_back(tmp_path):
    cfg = small_cfg(tmp_path)
    cfg.runtime.web_learning_rollback_on_fixed_regression = True
    cfg.runtime.web_learning_max_fixed_regression = 0.05
    cfg.runtime.replay_min_sequences = 2
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        from aletheia.web import Page
        # Seed an older lesson so replay has an eligible source.
        old_id = agent.memory.add(
            kind='lesson', source='seed', title='seed lesson', url='',
            text='Older language material about weather, travel, and careful observation. ' * 20,
        )
        agent.memory.replace_chunks(old_id, agent._memory_chunk_texts('Older language material about weather, travel, and careful observation. ' * 20))
        calls = iter([{'gain': 0.0}, {'gain': -0.20}, {'gain': 0.0}])
        agent.evaluate_fixed_reference = lambda: next(calls)
        original = agent.model.state_dict()['token_embedding.weight'].detach().clone()
        def fake_train(sequences, **kwargs):
            with torch.no_grad():
                agent.model.token_embedding.weight.add_(0.5)
            agent.last_replay_stats = {'requested': 2, 'used': 2, 'lesson_count': 1, 'lesson_ids': [old_id], 'per_epoch': [2]}
            return 2.0
        agent._train_sequence_collection = fake_train
        varied_external_text = ' '.join(
            f'New external material observation {index} describes rivers, weather, measurement, and learning.'
            for index in range(30)
        )
        page = Page(
            'https://example.org/new', 'new external page',
            varied_external_text,
            200, [],
        )
        result = agent._ingest(page, source='search', action_name='search')
        restored = agent.model.state_dict()['token_embedding.weight'].detach().clone()
        assert result['accepted'] is False
        assert result['replay_sequences_used'] == 2
        assert result['replay_lessons_used'] == 1
        assert torch.equal(original, restored)
        rows = agent.memory.recent(5, kinds=('external',))
        assert rows[-1]['action_name'] == 'search_rollback'
    finally:
        agent.close()


def test_cli_web_check_smoke(monkeypatch, tmp_path, capsys):
    import sys
    import aletheia.cli as cli
    from aletheia.web import Page

    class FakeWeb:
        last_errors = []
        def open(self, url):
            return Page(url, 'Example', 'hello world', 200, ['https://example.org/x'])
        def search(self, query, limit=5):
            return ['https://example.org/result']

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            from aletheia.config import load_config
            self.config = load_config(Path('config.json'))
            self.web = FakeWeb()
        def close(self):
            pass

    monkeypatch.setattr(cli, 'AletheiaAgent', FakeAgent)
    monkeypatch.setattr(sys, 'argv', ['aletheia', '--root', str(tmp_path), '--web-check', 'https://example.org'])
    cli.main()
    assert 'WEB_OK:' in capsys.readouterr().out


def test_rejected_experience_does_not_count_as_validated_lesson_or_replay(tmp_path):
    from aletheia.memory import ExperienceMemory
    memory = ExperienceMemory(tmp_path / "memory.sqlite3", 20, 20)
    try:
        rejected = memory.add(kind="lesson", source="bad", title="Rejected", url="", text="rejected text", accepted=False, replay_eligible=False)
        accepted = memory.add(kind="lesson", source="good", title="Accepted", url="", text="accepted text", accepted=True, replay_eligible=True)
        assert memory.lesson_count() == 1
        assert memory.experience_lesson_count() == 2
        assert [row["id"] for row in memory.recent_lessons(10)] == [accepted]
        all_lessons = memory.recent_lessons(10, validated_only=False)
        status = {row["id"]: (row["accepted"], row["replay_eligible"]) for row in all_lessons}
        assert status[rejected] == (0, 0)
        assert status[accepted] == (1, 1)
    finally:
        memory.close()


def test_rejected_guided_lesson_does_not_count_unique_tokens(tmp_path, monkeypatch):
    from pathlib import Path
    from aletheia.agent import AletheiaAgent
    from aletheia.config import load_config

    cfg = load_config(Path("config.json"))
    cfg.runtime.device = "cpu"
    cfg.runtime.teacher_epochs = 1
    cfg.runtime.teacher_max_chunks = 3
    cfg.runtime.teacher_sequence_length = 1
    cfg.runtime.replay_mix_fraction = 0.0
    cfg.runtime.replay_min_sequences = 0
    cfg.model.dim = 32
    cfg.model.layers = 1
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 64
    cfg.model.gradient_checkpointing = False
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        calls = {"n": 0}
        def fake_eval():
            calls["n"] += 1
            gain = 1.0 if calls["n"] == 1 else 0.0
            return {"available": True, "model_bpb": 7.0 - gain, "baseline_bpb": 7.0, "gain": gain, "reference_bytes": 80000}
        monkeypatch.setattr(agent, "evaluate_fixed_reference", fake_eval)
        result = agent.teach_text(
            Lesson(title="Rejected", source="test://rejected", text="This lesson should be rolled back. " * 40),
            epochs=1,
        )
        assert result["lesson_accepted"] is False
        assert result["unique_tokens_added"] == 0
        assert agent.total_tokens == 0
        row = agent.memory.recent_lessons(10, validated_only=False)[0]
        assert row["accepted"] == 0
        assert row["replay_eligible"] == 0
    finally:
        agent.close()


def test_dialogue_alignment_eval_deduplicates_responses(tmp_path):
    from pathlib import Path
    from aletheia.agent import AletheiaAgent
    from aletheia.config import load_config

    cfg = load_config(Path("config.json"))
    cfg.runtime.device = "cpu"
    cfg.model.dim = 32
    cfg.model.layers = 1
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 64
    cfg.model.gradient_checkpointing = False
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        pairs = [
            ("Turn A: Hi.", "Hello."),
            ("Turn A: Hello.", "Hello."),
            ("Turn A: How are you?", "I'm fine."),
        ]
        metrics = agent._alignment_eval(pairs, max_pairs=16)
        assert metrics["pairs"] == 2
        assert 0.0 <= metrics["top1"] <= 1.0
        assert 0.0 <= metrics["mrr"] <= 1.0
    finally:
        agent.close()


def test_guided_result_reports_multi_track_learning(tmp_path):
    from pathlib import Path
    from aletheia.agent import AletheiaAgent
    from aletheia.config import load_config

    cfg = load_config(Path("config.json"))
    cfg.runtime.device = "cpu"
    cfg.runtime.teacher_epochs = 1
    cfg.runtime.teacher_max_chunks = 4
    cfg.runtime.teacher_sequence_length = 1
    cfg.runtime.replay_mix_fraction = 0.0
    cfg.runtime.replay_min_sequences = 0
    cfg.model.dim = 32
    cfg.model.layers = 1
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 64
    cfg.model.gradient_checkpointing = False
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        result = agent.teach_text(Lesson(title="Multi-track", source="test://multi", text="A learner encounters language and learns from repeated context. " * 40), epochs=1)
        assert "local_improvement" in result
        assert "transfer_improvement" in result
        assert "aggregate_improvement" in result
        assert set(result["lesson_acceptance_tracks"]) == {"fixed_ok", "local_ok", "transfer_ok", "aggregate_ok"}
    finally:
        agent.close()



def test_dialogue_holdout_splits_whole_runs_and_is_reproducible():
    from aletheia.dialogue import split_dialogue_response_pairs
    runs = []
    for i in range(20):
        runs.append(
            f"A: question {i}?\nB: answer {i}.\n"
        )
    text = "\n".join(runs)
    train1, eval1 = split_dialogue_response_pairs(text, holdout_fraction=0.25, max_train_examples=0, max_eval_examples=0, seed=7)
    train2, eval2 = split_dialogue_response_pairs(text, holdout_fraction=0.25, max_train_examples=0, max_eval_examples=0, seed=7)
    assert train1 == train2 and eval1 == eval2
    train_responses = {response for _, response in train1}
    eval_responses = {response for _, response in eval1}
    assert train_responses.isdisjoint(eval_responses)
    assert train1 and eval1


def test_alignment_eval_reports_random_baselines(tmp_path):
    from aletheia.agent import AletheiaAgent
    from aletheia.config import load_config
    cfg = load_config(Path('config.json'))
    cfg.runtime.device = 'cpu'
    cfg.model.dim = 32
    cfg.model.layers = 1
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 64
    cfg.model.gradient_checkpointing = False
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        pairs = [(f'Turn A: topic {i}.', f'Response {i}.') for i in range(10)]
        metrics = agent._alignment_eval(pairs, max_pairs=5, seed=123)
        assert metrics['pairs'] == 5
        assert metrics['random_top1'] == pytest.approx(1 / 5)
        assert 0.0 < metrics['random_mrr'] <= 1.0
    finally:
        agent.close()


def test_reteach_creates_separate_attempt_and_preserves_prior_acceptance(tmp_path):
    from aletheia.agent import AletheiaAgent
    from aletheia.config import load_config
    cfg = load_config(Path('config.json'))
    cfg.runtime.device = 'cpu'
    cfg.runtime.teacher_epochs = 1
    cfg.runtime.teacher_max_chunks = 3
    cfg.runtime.teacher_sequence_length = 1
    cfg.runtime.replay_mix_fraction = 0.0
    cfg.runtime.replay_min_sequences = 0
    cfg.model.dim = 32
    cfg.model.layers = 1
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 64
    cfg.model.gradient_checkpointing = False
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        text = 'A stable learner records evidence and updates its model from repeated observations. ' * 30
        calls = iter([{'gain': 0.0}, {'gain': 0.0}, {'gain': 0.0}, {'gain': 0.0}])
        agent.evaluate_fixed_reference = lambda: next(calls)
        first = agent.teach_text(Lesson(title='same', source='same://lesson', text=text), epochs=1)
        assert first['lesson_accepted'] is True
        assert agent.memory.lesson_count() == 1
        # Force a rejected re-teach without allowing it to overwrite the accepted row.
        cfg.runtime.lesson_hard_fixed_regression = 0.01
        calls2 = iter([{'gain': 0.0}, {'gain': -0.1}, {'gain': -0.1}])
        agent.evaluate_fixed_reference = lambda: next(calls2)
        agent._train_sequence_collection = lambda *args, **kwargs: 9.0
        second = agent.teach_text(Lesson(title='same', source='same://lesson', text=text), epochs=1)
        assert second['lesson_accepted'] is False
        rows = agent.memory.recent_lessons(10, validated_only=False)
        statuses = [(row['accepted'], row['replay_eligible']) for row in rows]
        assert (1, 1) in statuses and (0, 0) in statuses
        assert agent.memory.lesson_count() == 1
    finally:
        agent.close()


def test_large_lesson_is_sharded_for_independent_acceptance(tmp_path, monkeypatch):
    from aletheia.agent import AletheiaAgent
    from aletheia.config import load_config
    cfg = load_config(Path('config.json'))
    cfg.runtime.device = 'cpu'
    cfg.runtime.teacher_shard_max_chars = 200
    cfg.runtime.teacher_epochs = 1
    cfg.runtime.teacher_max_chunks = 1
    cfg.runtime.teacher_sequence_length = 1
    cfg.runtime.replay_mix_fraction = 0.0
    cfg.runtime.replay_min_sequences = 0
    cfg.model.dim = 32
    cfg.model.layers = 1
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 64
    cfg.model.gradient_checkpointing = False
    path = tmp_path / 'large.txt'
    path.write_text('\n\n'.join(f'Paragraph {i}: distinct material for shard testing. ' * 4 for i in range(8)), encoding='utf-8')
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        monkeypatch.setattr(agent, 'evaluate_fixed_reference', lambda: {'gain': 0.0, 'model_bpb': 1.0, 'baseline_bpb': 1.0, 'reference_bytes': 80000})
        results = agent.teach_file(path, epochs=1)
        assert len(results) >= 2
        assert all('[shard ' in result['title'] for result in results)
    finally:
        agent.close()


def test_transfer_probe_samples_across_validated_history(tmp_path):
    from aletheia.agent import AletheiaAgent
    from aletheia.config import load_config
    cfg = load_config(Path('config.json'))
    cfg.runtime.device = 'cpu'
    cfg.runtime.lesson_transfer_documents = 4
    cfg.runtime.lesson_transfer_chunks_per_document = 1
    cfg.runtime.transfer_probe_candidate_pool = 64
    cfg.model.dim = 32
    cfg.model.layers = 1
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 64
    cfg.model.gradient_checkpointing = False
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        for i in range(8):
            rid = agent.memory.add(kind='lesson', source=f's{i}', title=f'lesson {i}', url='', text=(f'validated material number {i}. ' * 20), accepted=True, replay_eligible=True, merge_existing=False)
            agent.memory.replace_chunks(rid, agent._memory_chunk_texts(f'validated material number {i}. ' * 20, context=64))
        probe = agent._validated_transfer_probe()
        assert len(probe) >= 4
    finally:
        agent.close()


def test_eval_builder_excludes_reformatted_training_overlap(monkeypatch, tmp_path):
    import aletheia.eval_corpus as ec
    source = 'Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu.\n' * 20
    training = tmp_path / 'train.txt'
    training.write_text(source, encoding='utf-8')
    monkeypatch.setattr(ec, '_fetch', lambda url: ('Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu.\n\n' + 'Distinct external evaluation prose about a river, a school, and a scientist recording measurements. ' * 30) * 100)
    result = ec.build_reference(
        tmp_path / 'evaluation' / 'heldout.txt',
        target_bytes=50000,
        sources=[('source-a', 'https://example.org/a'), ('source-b', 'https://example.org/b')],
        exclude_paths=[str(training)],
    )
    text = (tmp_path / 'evaluation' / 'heldout.txt').read_text(encoding='utf-8').lower()
    assert 'distinct external evaluation prose' in text
    assert result['excluded_training_paths'] == [str(training.resolve())]


def test_domain_gain_can_offset_soft_fixed_regression(tmp_path, monkeypatch):
    from aletheia.agent import AletheiaAgent
    from aletheia.config import load_config
    cfg = load_config(Path('config.json'))
    cfg.runtime.device = 'cpu'
    cfg.runtime.teacher_epochs = 1
    cfg.runtime.teacher_max_chunks = 3
    cfg.runtime.teacher_sequence_length = 1
    cfg.runtime.replay_mix_fraction = 0.0
    cfg.runtime.replay_min_sequences = 0
    cfg.runtime.lesson_max_aggregate_regression = 0.03
    cfg.runtime.lesson_hard_fixed_regression = 0.25
    cfg.runtime.lesson_hard_local_regression = 0.20
    cfg.runtime.lesson_hard_transfer_regression = 0.20
    cfg.model.dim = 32
    cfg.model.layers = 1
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 64
    cfg.model.gradient_checkpointing = False
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        eval_calls = iter([{'gain': 0.0}, {'gain': -0.08}])
        monkeypatch.setattr(agent, 'evaluate_fixed_reference', lambda: next(eval_calls))
        # First call is before-holdout, second is after-holdout. Transfer is absent.
        holdout_calls = iter([2.0, 1.0])
        monkeypatch.setattr(agent, '_chunk_bpb', lambda chunks: next(holdout_calls))
        monkeypatch.setattr(agent, '_validated_transfer_probe', lambda exclude_ids=None: [])
        monkeypatch.setattr(agent, '_train_sequence_collection', lambda *args, **kwargs: 1.0)
        result = agent.teach_text(Lesson(title='offset', source='offset', text='Domain-specific material that is useful for this lesson. ' * 40), epochs=1)
        assert result['lesson_accepted'] is True
        assert result['lesson_fixed_gain_delta'] == pytest.approx(-0.08)
        assert result['local_improvement'] == pytest.approx(1.0)
    finally:
        agent.close()
