from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import AletheiaAgent
from .config import load_config


def _metric(value, digits=4):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _print_lessons(agent: AletheiaAgent) -> None:
    rows = agent.lessons(20)
    if not rows:
        print("No lessons learned yet.")
        return
    for row in rows:
        print(f"- {row['title']} | {row['source']} | progress={row['learning_progress']:.6f}")


def _print_status(agent: AletheiaAgent) -> None:
    print(
        agent.memory.summary(),
        "lessons=", agent.memory.lesson_count(),
        "experienced_lessons=", agent.memory.experience_lesson_count(),
        "unique_training_tokens=", agent.total_tokens,
        "validated_unique_training_tokens=", agent.total_tokens,
        "processed_training_tokens=", agent.processed_tokens,
        "optimizer_steps=", agent.training_steps,
        "parameters=", agent.parameter_count(),
        "context=", agent.model.cfg.context,
        "amp=", agent.amp_enabled,
        "amp_dtype=", str(agent.amp_dtype).replace("torch.", ""),
        "dialogue_examples=", agent.dialogue_examples_seen,
        "last_dialogue_accepted=", agent.last_dialogue_accepted,
        "last_dialogue_fixed_gain_delta=", agent.last_dialogue_fixed_gain_delta,
        "last_lesson_accepted=", agent.last_lesson_accepted,
        "last_lesson_fixed_gain_delta=", agent.last_lesson_fixed_gain_delta,
        "teacher_shard_max_chars=", agent.config.runtime.teacher_shard_max_chars,
        "replay_mix_fraction=", agent.config.runtime.replay_mix_fraction,
        "chat_generation_ready=", agent._chat_generation_ready(),
        "language_bpb=", agent.last_eval_bpb,
        "baseline_bpb=", agent.eval_baseline_bpb,
        "ready=", agent._language_ready(),
        "active_objective=", (agent.objectives.current().text if agent.objectives.current() else None),
        "identity_entities=", len(agent.perception.identities.entities),
        "visual_entities=", len(agent.perception.visual_entities.entities),
        "vision_tracks=", len(agent.perception.tracker.tracks),
        "vision_steps=", agent.perception.vision_steps,
        "audio_steps=", agent.perception.audio_steps,
        "speech_steps=", agent.speech.steps,
        "speech_loss=", agent.speech.last_loss,
        "speech_ready=", agent.speech.ready,
        "tts_steps=", agent.tts.steps,
        "tts_loss=", agent.tts.last_loss,
        "tts_holdout_loss=", agent.tts.last_holdout_loss,
        "tts_ready=", agent.tts.ready,
        "pretrain_shards_seen=", agent.pretrain_shards_seen,
        "pretrain_bytes_seen=", agent.pretrain_bytes_seen,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Aletheia from random initialization. Guided teaching uses only Aletheia's own model."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--cycles", type=int, default=25)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--teach", default=None, help="Teach from a URL, file, or directory. Prefix literal text with text:.")
    parser.add_argument("--pretrain", default=None, help="Run mixed foundation pretraining over a file or directory. No continual-learning acceptance veto is used.")
    parser.add_argument("--pretrain-epochs", type=int, default=None, help="Foundation pretraining passes over the interleaved corpus shards.")
    parser.add_argument("--pretrain-seed", type=int, default=None, help="Deterministic seed for mixed pretraining shard order.")
    parser.add_argument("--pretrain-max-bytes", type=int, default=None, help="Optional UTF-8 byte budget for streamed dataset pretraining (useful for large Parquet corpora).")
    parser.add_argument("--teach-parquet-max-bytes", type=int, default=None, help="Optional UTF-8 byte budget for direct Parquet teaching; large corpora should usually use --pretrain.")
    parser.add_argument("--teach-text", default=None, help="Teach literal text directly (alternative to --teach text:...).")
    parser.add_argument("--teach-stdin", action="store_true", help="Teach exact literal text read from stdin; preserves internal newlines and bypasses command-line length limits.")
    parser.add_argument("--tokenizer-train", default=None, help="Learn a from-scratch byte-fallback BPE tokenizer from a local text file or folder.")
    parser.add_argument("--tokenizer-vocab-size", type=int, default=None)
    parser.add_argument("--tokenizer-min-frequency", type=int, default=None)
    parser.add_argument("--tokenizer-max-bytes", type=int, default=None)
    parser.add_argument("--teach-epochs", type=int, default=None)
    parser.add_argument("--crawl-pages", type=int, default=None, help="Override the same-domain page limit for --teach URLs.")
    parser.add_argument("--crawl-depth", type=int, default=None, help="Override the same-domain crawl depth for --teach URLs.")
    parser.add_argument("--crawl-attempts", type=int, default=None, help="Override the maximum failed/successful URL attempts for --teach URLs.")
    parser.add_argument("--dialogue", action="store_true", help="Treat supplied text as structural dialogue and train turn pairs.")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--message", default=None)
    parser.add_argument("--lessons", action="store_true")
    parser.add_argument("--status", action="store_true", help="Print current learning, evaluation, replay, and readiness state.")
    parser.add_argument("--reset-learning", action="store_true")
    parser.add_argument("--build-eval-corpus", action="store_true", help="Build the fixed external evaluation corpus before learning.")
    parser.add_argument("--web-check", default=None, help="Open one public HTTP(S) URL and report whether Aletheia's web layer can fetch and parse it.")
    parser.add_argument("--web-search", default=None, help="Run one configured public web search and report the discovered URLs.")
    parser.add_argument("--browser-open", default=None, help="Open a public HTTP(S) URL in Aletheia's interactive Chromium environment and inspect DOM + screenshot.")
    parser.add_argument("--browser-action", default=None, help="Execute one generic browser action as a JSON object against the current browser observation.")
    parser.add_argument("--browser-status", action="store_true", help="Print the current interactive browser environment state.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--goal", default=None, help="Set a persistent objective for autonomous learning.")
    parser.add_argument("--goal-verifier", choices=("manual", "threshold", "human", "answer", "unit_test"), default=None, help="Verification contract for --goal.")
    parser.add_argument("--goal-threshold", type=float, default=None, help="Completion threshold for threshold-verified goals.")
    parser.add_argument("--goal-consecutive", type=int, default=None, help="Required consecutive verified outcomes before automatic completion.")
    parser.add_argument("--plan", action="store_true", help="Show the current model-based objective plan and latent rollout.")
    parser.add_argument("--see", default=None, help="Observe an image with Aletheia's randomly initialized visual system.")
    parser.add_argument("--alias", default="", help="Optional alias attached to a newly observed visual entity.")
    parser.add_argument("--recognize", default=None, help="Recognize an image against learned visual identity memory.")
    parser.add_argument("--hear", default=None, help="Encode a WAV file with Aletheia's from-scratch audio representation.")
    parser.add_argument("--speech-train", default=None, help="Train the from-scratch speech-to-byte CTC bridge from a WAV/transcript manifest.")
    parser.add_argument("--speech-epochs", type=int, default=1)
    parser.add_argument("--speech-max-examples", type=int, default=0)
    parser.add_argument("--transcribe", default=None, help="Transcribe a WAV using Aletheia's learned speech bridge when it is ready.")
    parser.add_argument("--enroll-speaker", default=None, help="Enroll a voice sample into persistent from-scratch speaker identity memory.")
    parser.add_argument("--speaker-alias", default="", help="Alias used with --enroll-speaker.")
    parser.add_argument("--recognize-speaker", default=None, help="Identify a WAV against learned speaker identity memory.")
    parser.add_argument("--listen", type=float, default=None, help="Listen to the microphone for the given number of seconds, incrementally transcribe, identify the speaker, and respond.")
    parser.add_argument("--speak", default=None, help="Speak text using the host speech backend.")
    parser.add_argument("--tts-train", default=None, help="Train Aletheia's from-scratch speaker-conditioned TTS from a paired audio/text manifest.")
    parser.add_argument("--tts-epochs", type=int, default=1)
    parser.add_argument("--tts-max-examples", type=int, default=0)
    parser.add_argument("--synthesize", default=None, help="Synthesize speech with Aletheia's learned TTS.")
    parser.add_argument("--voice-sample", default=None, help="WAV reference used as the learned speaker conditioning for --synthesize.")
    parser.add_argument("--voice-speaker", default="", help="Enrolled speaker alias or id used as TTS conditioning.")
    parser.add_argument("--voice-output", default=None, help="Output WAV path for learned TTS.")
    parser.add_argument("--voice-play", action="store_true", help="Play learned TTS output through the local audio device.")
    parser.add_argument("--camera", type=int, default=None, help="Open a local camera device and learn from live visual observations.")
    parser.add_argument("--camera-frames", type=int, default=0, help="Frames to process; 0 means continuous until stopped.")
    parser.add_argument("--camera-show", action="store_true", help="Display the camera preview while learning.")
    parser.add_argument("--companion", action="store_true", help="Run local camera+microphone companion mode; Aletheia may initiate speech after learned readiness.")
    parser.add_argument("--companion-seconds", type=int, default=None, help="Optional duration for --companion; 0 means until interrupted.")
    parser.add_argument("--forever", action="store_true", help="Run the autonomous developmental loop until interrupted.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between autonomous cycles in --forever mode.")
    parser.add_argument("--evolve", action="store_true", help="Run one self-improvement experiment on a developmental parameter.")
    parser.add_argument("--evolve-generation", action="store_true", help="Clone the full learned state into a candidate population and benchmark successors; promotion is staged for restart when enabled.")
    parser.add_argument("--evolution-population", type=int, default=None, help="Override the generational candidate population size.")
    parser.add_argument("--evolution-no-promote", action="store_true", help="Evaluate a generational candidate population without promoting a winner.")
    parser.add_argument("--evolution-status", action="store_true", help="Show the active generation and recent evolution lineage.")
    parser.add_argument("--evolve-code", default=None, help="Ask Aletheia to propose and validate an allowlisted source rewrite.")
    parser.add_argument("--approve-evolve-code", action="store_true", help="Approve the currently staged validated source rewrite.")
    parser.add_argument("--evolve-instruction", default="improve the target module without changing unrelated behavior", help="Instruction for --evolve-code.")
    parser.add_argument("--learning-add", default=None, help="Queue a supported local file for source-separated extraction, evaluation and guarded learning.")
    parser.add_argument("--learning-add-text", default=None, help="Queue literal text for learning; long text is better passed in a .txt file with --learning-add.")
    parser.add_argument("--learning-teach-file", default=None, help="Queue a supported file and immediately run extraction, holdout evaluation and guarded learning in one command.")
    parser.add_argument("--learning-teach-text", default=None, help="Queue literal text and immediately run the guarded learning pipeline.")
    parser.add_argument("--learning-process-queue", action="store_true", help="Run queued jobs in order with one Aletheia process.")
    parser.add_argument("--learning-max-jobs", type=int, default=25, help="Maximum jobs processed by --learning-process-queue (default: 25).")
    parser.add_argument("--learning-max-train-chars", type=int, default=250000, help="Per-job training text budget to bound local compute (default: 250000).")
    parser.add_argument("--learning-max-eval-chars", type=int, default=100000, help="Per-job held-out evaluation text budget (default: 100000).")
    parser.add_argument("--learning-title", default=None, help="Human-readable title for text or file provenance.")
    parser.add_argument("--learning-license", default="", help="Data license or usage permission supplied by the user (recorded as provenance).")
    parser.add_argument("--learning-source-uri", default="", help="Original URL or source reference recorded as provenance; it is not fetched automatically.")
    parser.add_argument("--learning-jobs", action="store_true", help="List recent persistent learning jobs.")
    parser.add_argument("--learning-job-status", default=None, help="Show one learning job by ID.")
    parser.add_argument("--learning-run-job", default=None, help="Run a queued learning job by ID.")
    parser.add_argument("--learning-run-next", action="store_true", help="Run the oldest queued learning job.")
    parser.add_argument("--learning-retry-job", default=None, help="Requeue a failed/rejected job that has a durable train/evaluation split.")
    parser.add_argument("--learning-max-regression-bpb", type=float, default=0.08, help="Maximum permitted source-holdout regression in bits/byte before restoring prior weights.")
    parser.add_argument("--learning-job-epochs", type=int, default=None, help="Optional epoch override for queued learning jobs.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    learning_actions = [
        args.learning_add is not None,
        args.learning_add_text is not None,
        args.learning_teach_file is not None,
        args.learning_teach_text is not None,
        bool(args.learning_process_queue),
        bool(args.learning_jobs),
        args.learning_job_status is not None,
        args.learning_run_job is not None,
        bool(args.learning_run_next),
        args.learning_retry_job is not None,
    ]
    if sum(bool(value) for value in learning_actions) > 1:
        parser.error("choose only one --learning-* queue action per invocation")
    if any(learning_actions):
        from .learning_pipeline import LearningPipeline
        with LearningPipeline(
            root,
            max_training_chars=max(64, int(args.learning_max_train_chars)),
            max_evaluation_chars=max(64, int(args.learning_max_eval_chars)),
        ) as queue:
            if args.learning_add is not None:
                result = queue.enqueue_file(
                    Path(args.learning_add), license_name=args.learning_license,
                    source_uri=args.learning_source_uri, title=args.learning_title,
                )
                print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            elif args.learning_add_text is not None:
                result = queue.enqueue_text(
                    args.learning_add_text,
                    title=args.learning_title or "pasted-text.txt",
                    license_name=args.learning_license or "user-provided",
                    source_uri=args.learning_source_uri,
                )
                print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            elif args.learning_jobs:
                print(json.dumps(queue.list_jobs(), indent=2, ensure_ascii=False, default=str))
            elif args.learning_job_status is not None:
                print(json.dumps(queue.get_job(args.learning_job_status), indent=2, ensure_ascii=False, default=str))
            elif args.learning_retry_job is not None:
                print(json.dumps(queue.retry(args.learning_retry_job), indent=2, ensure_ascii=False, default=str))
            else:
                config = load_config(Path(args.config))
                if args.device is not None:
                    config.runtime.device = args.device
                agent = AletheiaAgent(root, config)
                try:
                    if args.learning_teach_file is not None:
                        queued = queue.enqueue_file(
                            Path(args.learning_teach_file), license_name=args.learning_license,
                            source_uri=args.learning_source_uri, title=args.learning_title,
                        )
                        result = queue.run_job(queued["id"], agent, max_regression_bpb=args.learning_max_regression_bpb, epochs=args.learning_job_epochs) if queued.get("state") == "queued" else queued
                        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
                    elif args.learning_teach_text is not None:
                        queued = queue.enqueue_text(
                            args.learning_teach_text, title=args.learning_title or "pasted-text.txt",
                            license_name=args.learning_license or "user-provided", source_uri=args.learning_source_uri,
                        )
                        result = queue.run_job(queued["id"], agent, max_regression_bpb=args.learning_max_regression_bpb, epochs=args.learning_job_epochs) if queued.get("state") == "queued" else queued
                        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
                    elif args.learning_process_queue:
                        outcomes = []
                        cap = max(1, min(500, int(args.learning_max_jobs)))
                        for _ in range(cap):
                            outcome = queue.run_next(agent, max_regression_bpb=args.learning_max_regression_bpb, epochs=args.learning_job_epochs)
                            if outcome is None:
                                break
                            outcomes.append(outcome)
                        print(json.dumps({"processed": len(outcomes), "jobs": outcomes}, indent=2, ensure_ascii=False, default=str))
                    elif args.learning_run_job is not None:
                        result = queue.run_job(
                            args.learning_run_job, agent,
                            max_regression_bpb=args.learning_max_regression_bpb,
                            epochs=args.learning_job_epochs,
                        )
                        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
                    else:
                        result = queue.run_next(
                            agent,
                            max_regression_bpb=args.learning_max_regression_bpb,
                            epochs=args.learning_job_epochs,
                        )
                        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
                finally:
                    agent.close()
        return

    if args.build_eval_corpus:
        config = load_config(Path(args.config))
        target_root = root.resolve()
        from .eval_corpus import build_reference
        eval_path = (target_root / config.runtime.language_eval_path).resolve()
        result = build_reference(
            eval_path,
            target_bytes=int(config.runtime.evaluation_corpus_target_bytes),
            exclude_paths=list(config.runtime.evaluation_exclude_paths or []),
            overlap_words=int(config.runtime.evaluation_overlap_words),
            overlap_match_fraction=float(config.runtime.evaluation_overlap_match_fraction),
        )
        print(f"Built evaluation corpus: {result['bytes']} bytes from {len(result['sources'])} prose sources")
        for failure in result.get("errors", []):
            print(f"EVAL_SOURCE_ERROR: {failure['title']} | {failure['url']} | {failure['error']}")
        if not (args.reset_learning or args.teach or args.pretrain or args.chat or args.message is not None or args.lessons or args.status or args.plan or args.web_check or args.web_search or args.goal or args.goal_verifier or args.goal_threshold is not None or args.goal_consecutive is not None or args.see or args.recognize or args.hear or args.speak or args.speech_train or args.tts_train or args.transcribe or args.synthesize is not None or args.enroll_speaker or args.recognize_speaker or args.listen is not None or args.camera is not None or args.companion or args.evolve or args.evolve_generation or args.evolution_status or args.evolve_code or args.approve_evolve_code or args.teach_stdin or args.tokenizer_train):
            return

    if args.reset_learning:
        archive = AletheiaAgent.reset_storage(root)
        if archive:
            print(f"Archived previous learning to: {archive}")
        else:
            print("No prior learned state found.")
        # Reset is a destructive state-management operation, not an implicit request
        # to launch the autonomous development loop. Continue only when the caller
        # explicitly paired reset with another operation such as --teach/--status.
        if not (args.build_eval_corpus or args.teach is not None or args.pretrain is not None or args.teach_text is not None
                or args.chat or args.message is not None or args.lessons or args.status or args.plan
                or args.web_check or args.web_search or args.goal or args.goal_verifier or args.goal_threshold is not None or args.goal_consecutive is not None or args.plan or args.see or args.recognize or args.hear
                or args.speak or args.speech_train or args.tts_train or args.transcribe or args.synthesize is not None or args.enroll_speaker or args.recognize_speaker or args.listen is not None or args.camera is not None or args.companion or args.evolve or args.evolve_code or args.forever
                or args.browser_open is not None or args.browser_action is not None or args.browser_status or args.teach_stdin or args.tokenizer_train):
            return

    config = load_config(Path(args.config))
    if args.device is not None:
        config.runtime.device = args.device
    if args.crawl_pages is not None:
        config.runtime.web_crawl_max_pages = max(1, int(args.crawl_pages))
    if args.crawl_depth is not None:
        config.runtime.web_crawl_max_depth = max(0, int(args.crawl_depth))
    if args.crawl_attempts is not None:
        config.runtime.web_crawl_max_attempts = max(1, int(args.crawl_attempts))
    if args.pretrain_max_bytes is not None:
        config.runtime.pretrain_max_source_bytes = max(0, int(args.pretrain_max_bytes))
    if args.teach_parquet_max_bytes is not None:
        config.runtime.teach_parquet_max_source_bytes = max(0, int(args.teach_parquet_max_bytes))
    agent = AletheiaAgent(root, config)
    try:
        if args.tokenizer_train is not None:
            print("TOKENIZER_TRAIN:", agent.train_tokenizer(args.tokenizer_train, vocab_size=args.tokenizer_vocab_size, min_frequency=args.tokenizer_min_frequency, max_training_bytes=args.tokenizer_max_bytes))
        elif args.teach_stdin:
            content = sys.stdin.read()
            print("TAUGHT:", agent.teach_text_exact(content, title="stdin text", source="teacher://stdin"))
        elif args.goal is not None:
            result = agent.set_objective(
                args.goal,
                verifier_type=args.goal_verifier,
                completion_threshold=args.goal_threshold,
                required_consecutive=args.goal_consecutive,
            )
            print(f"OBJECTIVE: {result['id']} | {result['text']}")
        elif args.plan:
            print("PLAN:", json.dumps(agent.planning_status(), indent=2, ensure_ascii=False, default=str))
        elif args.see is not None:
            result = agent.observe_image(args.see, alias=args.alias)
            print("VISION:", result)
        elif args.recognize is not None:
            result = agent.recognize_image(args.recognize)
            print("RECOGNITION:", result if result else "no visual identity memory yet")
        elif args.hear is not None:
            print("AUDIO:", agent.hear_audio(args.hear))
        elif args.speech_train is not None:
            print("SPEECH_TRAIN:", agent.train_speech(args.speech_train, epochs=args.speech_epochs, max_examples=args.speech_max_examples))
        elif args.tts_train is not None:
            print("TTS_TRAIN:", agent.train_tts(args.tts_train, epochs=args.tts_epochs, max_examples=args.tts_max_examples))
        elif args.synthesize is not None:
            print(
                "SYNTHESIS:",
                agent.synthesize_voice(
                    args.synthesize,
                    speaker_sample=args.voice_sample,
                    speaker=args.voice_speaker,
                    output=args.voice_output,
                    play=args.voice_play,
                ),
            )
        elif args.transcribe is not None:
            if not agent.speech.ready:
                print("TRANSCRIBE_NOT_READY: train a paired WAV/transcript manifest first")
            else:
                print("TRANSCRIPT:", agent.transcribe_audio(args.transcribe))
        elif args.enroll_speaker is not None:
            print("SPEAKER_ENROLL:", agent.enroll_speaker(args.enroll_speaker, args.speaker_alias))
        elif args.recognize_speaker is not None:
            print("SPEAKER_RECOGNITION:", agent.identify_speaker(args.recognize_speaker))
        elif args.listen is not None:
            print("LISTEN:", agent.listen_microphone(seconds=args.listen))
        elif args.speak is not None:
            print("SPEECH:", agent.speak(args.speak))
        elif args.camera is not None:
            result = agent.see_camera(device_index=args.camera, frames=args.camera_frames, alias=args.alias, show=args.camera_show)
            print("CAMERA:", {"frames": len(result), "last": result[-1] if result else None})
        elif args.companion:
            agent.run_companion(camera_index=0 if args.camera is None else args.camera, max_seconds=args.companion_seconds, show=args.camera_show)
        elif args.evolution_status:
            print("EVOLUTION_STATUS:", json.dumps(agent.evolution_status(), indent=2, ensure_ascii=False, default=str))
        elif args.evolve_generation:
            result = agent.evolve_generation(
                population_size=args.evolution_population,
                auto_promote=False if args.evolution_no_promote else None,
            )
            print("EVOLUTION_GENERATION:", json.dumps(result, indent=2, ensure_ascii=False, default=str))
        elif args.evolve:
            result = agent._cap_evolve()[1]
            print("EVOLUTION:", result)
        elif args.approve_evolve_code:
            print("SOURCE_EVOLUTION_APPROVAL:", agent.evolution.approve_pending_source_change())
        elif args.evolve_code is not None:
            result = agent.evolve_source(args.evolve_code, args.evolve_instruction)
            print("SOURCE_EVOLUTION:", result)
        elif args.web_check is not None:
            page = agent.web.open(args.web_check)
            print(f"WEB_OK: status={page.status_code} | title={page.title[:120]} | chars={len(page.text)} | links={len(page.links)} | url={page.url}")
        elif args.web_search is not None:
            links = agent.web.search(args.web_search, config.runtime.search_result_limit)
            if links:
                print(f"WEB_SEARCH_OK: query={args.web_search!r} | results={len(links)}")
                for link in links:
                    print(link)
            else:
                print(f"WEB_SEARCH_EMPTY: query={args.web_search!r}")
                for error in agent.web.last_errors:
                    print(f"WEB_SEARCH_ERROR: {error}")
        elif args.browser_open is not None:
            print("BROWSER_OPEN:", json.dumps(agent.browser_open(args.browser_open), indent=2, ensure_ascii=False, default=str))
        elif args.browser_action is not None:
            action = json.loads(args.browser_action)
            if not isinstance(action, dict):
                raise ValueError("--browser-action requires a JSON object")
            print("BROWSER_ACTION:", json.dumps(agent.browser_action(action), indent=2, ensure_ascii=False, default=str))
        elif args.browser_status:
            print("BROWSER_STATUS:", json.dumps(agent.browser_status(), indent=2, ensure_ascii=False, default=str))
        elif args.forever:
            agent.run_forever(args.train_steps, sleep_seconds=args.sleep)
        elif args.pretrain is not None:
            results = agent.pretrain(
                Path(args.pretrain),
                epochs=args.pretrain_epochs,
                seed=args.pretrain_seed,
            )
            for result in results:
                fixed = result.get("fixed_eval_gain")
                fixed_text = "n/a" if fixed is None else f"{float(fixed):.4f}"
                cleaning = result.get("transcript_cleaning", {})
                print(
                    f"PRETRAIN: epoch {result['epoch']}/{result['epochs']} | shard {result['shard']}/{result['shards']} "
                    f"| {result['title']} | chars={result['chars']} | local_gain={result['learning_progress']:.6f} "
                    f"| fixed_gain={fixed_text} | loss={result['train_loss']:.4f} "
                    f"| steps={result['optimizer_steps']} | processed_tokens={result['processed_training_tokens']} "
                    f"| transcript={bool(cleaning.get('transcript_detected', False))} "
                    f"| removed_chars={cleaning.get('removed_chars', 0)}"
                )
        elif args.teach is not None or args.teach_text is not None:
            teaching_source = args.teach if args.teach is not None else "text:" + args.teach_text
            results = agent.teach_input(teaching_source, epochs=args.teach_epochs, dialogue=args.dialogue)
            for result in results:
                print(
                    f"TAUGHT: {result['title']} | chars={result['chars']} | chunks={result['chunks']} "
                    f"| loss={result['loss_before']:.4f}->{result['loss_after']:.4f} "
                    f"| learning_progress={result['learning_progress']:.6f} | holdout_bpb={_metric(result.get('holdout_bpb_after'))} "
                    f"| fixed_gain={_metric(result.get('fixed_eval_gain'))} "
                    f"| accepted={result.get('lesson_accepted', True)} "
                    f"| aggregate={_metric(result.get('aggregate_improvement'))} "
                    f"| replay={result.get('replay_sequences_used', 0)} seq/{result.get('replay_lessons_used', 0)} lessons "
                    f"| dialogue_examples={result.get('dialogue_examples', 0)}"
                )
                for skipped in result.get('skipped_files', []):
                    print(f"SKIPPED: {skipped}")
        elif args.lessons:
            _print_lessons(agent)
        elif args.status:
            _print_status(agent)
        elif args.plan:
            print(json.dumps(agent.planning_status(), indent=2, ensure_ascii=False, default=str))
        elif args.message is not None:
            print(agent.communicate(args.message, max_tokens=config.runtime.chat_max_response_tokens))
        elif args.chat:
            print("Aletheia developmental chat. Commands: /learn, /paste, /endpaste, /goal, /plan, /good, /bad, /see, /recognize, /hear, /transcribe, /enroll-voice, /recognize-voice, /listen, /speak, /tts-train, /synthesize, /camera, /companion, /browser-open, /browser-action, /browser-status, /evolve, /evolve-generation, /evolution-status, /forever, /lessons, /status, /reset, /quit.")
            while True:
                message = input("YOU> ").strip()
                if not message:
                    continue
                if message == "/quit":
                    break
                if message == "/lessons":
                    _print_lessons(agent)
                    continue
                if message == "/status":
                    _print_status(agent)
                    continue
                if message == "/plan":
                    try:
                        print("PLAN>", json.dumps(agent.planning_status(), indent=2, ensure_ascii=False, default=str))
                    except Exception as exc:
                        print("PLAN_ERROR>", repr(exc))
                    continue
                if message == "/reset":
                    agent.close()
                    archive = AletheiaAgent.reset_storage(root)
                    print(f"Learning reset. Archive: {archive}")
                    agent = AletheiaAgent(root, config)
                    continue
                if message.startswith("/goal "):
                    try:
                        print("OBJECTIVE>", agent.set_objective(message[len("/goal "):].strip()))
                    except Exception as exc:
                        print("GOAL_ERROR>", repr(exc))
                    continue
                if message == "/good" or message.startswith("/good "):
                    print("FEEDBACK>", agent.objective_feedback(True, message[5:].strip()))
                    continue
                if message == "/bad" or message.startswith("/bad "):
                    print("FEEDBACK>", agent.objective_feedback(False, message[4:].strip()))
                    continue
                if message.startswith("/browser-open "):
                    try:
                        print("BROWSER>", json.dumps(agent.browser_open(message[len("/browser-open "):].strip()), indent=2, ensure_ascii=False, default=str))
                    except Exception as exc:
                        print("BROWSER_ERROR>", repr(exc))
                    continue
                if message.startswith("/browser-action "):
                    try:
                        action = json.loads(message[len("/browser-action "):].strip())
                        if not isinstance(action, dict):
                            raise ValueError("browser action must be a JSON object")
                        print("BROWSER>", json.dumps(agent.browser_action(action), indent=2, ensure_ascii=False, default=str))
                    except Exception as exc:
                        print("BROWSER_ERROR>", repr(exc))
                    continue
                if message == "/browser-status":
                    try:
                        print("BROWSER>", json.dumps(agent.browser_status(), indent=2, ensure_ascii=False, default=str))
                    except Exception as exc:
                        print("BROWSER_ERROR>", repr(exc))
                    continue
                if message.startswith("/see "):
                    try:
                        print("VISION>", agent.observe_image(message[len("/see "):].strip()))
                    except Exception as exc:
                        print("VISION_ERROR>", repr(exc))
                    continue
                if message.startswith("/recognize "):
                    try:
                        print("RECOGNITION>", agent.recognize_image(message[len("/recognize "):].strip()))
                    except Exception as exc:
                        print("RECOGNITION_ERROR>", repr(exc))
                    continue
                if message.startswith("/hear "):
                    try:
                        print("AUDIO>", agent.hear_audio(message[len("/hear "):].strip()))
                    except Exception as exc:
                        print("AUDIO_ERROR>", repr(exc))
                    continue
                if message.startswith("/enroll-voice "):
                    parts = message[len("/enroll-voice "):].strip().split(" ", 1)
                    if len(parts) != 2:
                        print("SPEAKER_ENROLL_ERROR> usage: /enroll-voice <wav> <alias>")
                    else:
                        try:
                            print("SPEAKER_ENROLL>", agent.enroll_speaker(parts[0], parts[1]))
                        except Exception as exc:
                            print("SPEAKER_ENROLL_ERROR>", repr(exc))
                    continue
                if message.startswith("/recognize-voice "):
                    try:
                        print("SPEAKER_RECOGNITION>", agent.identify_speaker(message[len("/recognize-voice "):].strip()))
                    except Exception as exc:
                        print("SPEAKER_RECOGNITION_ERROR>", repr(exc))
                    continue
                if message.startswith("/listen"):
                    parts = message.split()
                    try:
                        seconds = float(parts[1]) if len(parts) > 1 else 5.0
                        print("LISTEN>", agent.listen_microphone(seconds=seconds))
                    except Exception as exc:
                        print("LISTEN_ERROR>", repr(exc))
                    continue
                if message.startswith("/transcribe "):
                    try:
                        source = message[len("/transcribe "):].strip()
                        if not agent.speech.ready:
                            print("TRANSCRIBE_NOT_READY> train a paired WAV/transcript manifest until holdout CER passes")
                        else:
                            print("TRANSCRIPT>", agent.transcribe_audio(source))
                    except Exception as exc:
                        print("TRANSCRIBE_ERROR>", repr(exc))
                    continue
                if message.startswith("/camera"):
                    parts = message.split()
                    device_index = int(parts[1]) if len(parts) > 1 else 0
                    frames = int(parts[2]) if len(parts) > 2 else 0
                    try:
                        print("CAMERA>", agent.see_camera(device_index=device_index, frames=frames, alias="", show=False)[-1:])
                    except Exception as exc:
                        print("CAMERA_ERROR>", repr(exc))
                    continue
                if message == "/companion":
                    try:
                        agent.run_companion(camera_index=0, show=False)
                    except Exception as exc:
                        print("COMPANION_ERROR>", repr(exc))
                    continue
                if message.startswith("/tts-train "):
                    try:
                        print("TTS_TRAIN>", agent.train_tts(message[len("/tts-train "):].strip()))
                    except Exception as exc:
                        print("TTS_TRAIN_ERROR>", repr(exc))
                    continue
                if message.startswith("/synthesize "):
                    try:
                        print("SYNTHESIS>", agent.synthesize_voice(message[len("/synthesize "):].strip(), play=False))
                    except Exception as exc:
                        print("SYNTHESIS_ERROR>", repr(exc))
                    continue
                if message.startswith("/speak "):
                    try:
                        print("SPEECH>", agent.speak(message[len("/speak "):].strip()))
                    except Exception as exc:
                        print("SPEECH_ERROR>", repr(exc))
                    continue
                if message.startswith("/evolve-code "):
                    parts = message[len("/evolve-code "):].strip().split(" ", 1)
                    if len(parts) != 2:
                        print("EVOLUTION_CODE_ERROR> usage: /evolve-code <relative.py> <instruction>")
                    else:
                        try:
                            print("SOURCE_EVOLUTION>", agent.evolve_source(parts[0], parts[1]))
                        except Exception as exc:
                            print("EVOLUTION_CODE_ERROR>", repr(exc))
                    continue
                if message == "/evolution-status":
                    try:
                        print("EVOLUTION_STATUS>", json.dumps(agent.evolution_status(), indent=2, ensure_ascii=False, default=str))
                    except Exception as exc:
                        print("EVOLUTION_STATUS_ERROR>", repr(exc))
                    continue
                if message.startswith("/evolve-generation"):
                    try:
                        parts = message.split()
                        population = int(parts[1]) if len(parts) > 1 else None
                        print("EVOLUTION_GENERATION>", json.dumps(agent.evolve_generation(population_size=population), indent=2, ensure_ascii=False, default=str))
                    except Exception as exc:
                        print("EVOLUTION_GENERATION_ERROR>", repr(exc))
                    continue
                if message == "/evolve":
                    try:
                        print("EVOLUTION>", agent._cap_evolve()[1])
                    except Exception as exc:
                        print("EVOLUTION_ERROR>", repr(exc))
                    continue
                if message == "/forever":
                    agent.run_forever()
                    continue
                if message == "/paste":
                    print("PASTE MODE: enter /endpaste on its own line to finish. Internal newlines are preserved.")
                    lines = []
                    while True:
                        try:
                            line = input("PASTE> ")
                        except EOFError:
                            break
                        if line == "/endpaste":
                            break
                        lines.append(line)
                    try:
                        result = agent.teach_text_exact("\n".join(lines), title="pasted text", source="teacher://paste")
                        print("TAUGHT>", result)
                    except Exception as exc:
                        print("TEACH_ERROR>", repr(exc))
                    continue
                if message.startswith("/learn "):
                    source = message[len("/learn "):].strip()
                    dialogue = False
                    if source.startswith("--dialogue "):
                        dialogue = True
                        source = source[len("--dialogue "):].strip()
                    try:
                        results = agent.teach_input(source, dialogue=dialogue)
                        for result in results:
                            print(
                                f"TAUGHT> {result['title']} | progress={result['learning_progress']:.6f} "
                                f"| loss={result['loss_before']:.4f}->{result['loss_after']:.4f}"
                            )
                            for skipped in result.get('skipped_files', []):
                                print(f"SKIPPED> {skipped}")
                    except Exception as exc:
                        print("TEACH_ERROR>", repr(exc))
                    continue
                print("ALETHEIA>", agent.communicate(message, max_tokens=config.runtime.chat_max_response_tokens))
        else:
            agent.run(args.cycles, args.train_steps)
    finally:
        agent.close()
