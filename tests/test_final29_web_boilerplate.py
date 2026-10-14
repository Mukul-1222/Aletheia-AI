from aletheia.web import WebExplorer


def test_html_extraction_removes_common_widget_chrome_but_keeps_article():
    html = """
    <html><body>
      <div class="cookie-consent">Accept all cookies</div>
      <div class="related-articles"><p>Related articles</p><p>Other story</p></div>
      <article><h1>Learning systems</h1>
      <p>Learning systems update parameters from repeated observations and measured prediction errors.</p>
      <p>The evaluation should use data that was not present in training.</p></article>
      <div class="share-toolbar">Share this article</div>
    </body></html>
    """
    text = WebExplorer.extract_text(html)
    assert "Learning systems" in text
    assert "update parameters" in text
    assert "not present in training" in text
    assert "Accept all cookies" not in text
    assert "Other story" not in text
    assert "Share this article" not in text


def test_boilerplate_phrase_filter_only_removes_standalone_ui_segments():
    text = "Skip to content. A model learns from examples. Read more. Its result must be evaluated."
    cleaned = WebExplorer._remove_boilerplate_phrases(text)
    assert "Skip to content" not in cleaned
    assert "Read more" not in cleaned
    assert "A model learns from examples" in cleaned
    assert "Its result must be evaluated" in cleaned


def test_web_boilerplate_filter_preserves_legitimate_mentions_in_prose():
    text = "The tutorial explains why users may need to sign in before changing account settings."
    assert WebExplorer._remove_boilerplate_phrases(text) == text
