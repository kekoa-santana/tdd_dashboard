"""Markup contracts for the responsive Player Profile hero."""
from __future__ import annotations

from views.player_profile import _editorial_hero_html, _section_head


def _hero(**overrides) -> str:
    values = {
        "name": 'Jane "Slugger" <X>',
        "team": "NYM",
        "header_parts": ["Age 27", "Bats L", "Power-Speed", "<script>alert(1)</script>"],
        "diamond_html": '<span class="diamonds">rating</span>',
        "tools_html": '<span class="tools">tools</span>',
        "injury_html": "",
        "vitals": [("21.4%", "Proj K%", "-1.2pp")],
        "scouting_text": "Patient hitter.",
        "player_id": 660271,
    }
    values.update(overrides)
    return _editorial_hero_html(**values)


def test_editorial_hero_uses_responsive_image_contract():
    html = _hero()

    assert 'data-page="player-profile"' in html
    assert 'class="portrait-img"' in html
    assert "srcset=" in html
    assert 'sizes="(max-width: 479px) 32vw' in html
    assert 'style="width:100%;height:100%' not in html
    assert 'class="initials portrait-fallback"' in html


def test_editorial_hero_renders_trusted_team_markup_and_escapes_data():
    html = _hero()

    assert '<span class="tdd-team-abbr" data-team="NYM">New York Mets</span>' in html
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in html
    assert 'alt="Jane &quot;Slugger&quot; &lt;X&gt;"' in html
    assert '<span class="diamonds">rating</span>' in html


def test_editorial_hero_keeps_initials_when_no_headshot_is_available():
    html = _hero(name="Sample Player", player_id=None)

    assert 'class="initials portrait-fallback"' in html
    assert ">SP</span>" in html
    assert 'class="portrait-img"' not in html


def test_section_header_is_balanced_and_self_contained():
    html = _section_head("Season Stats", "Observed < projected")

    assert html.startswith('<section class="p-section">')
    assert html.endswith("</section>")
    assert html.count("<section") == html.count("</section>") == 1
    assert "Observed &lt; projected" in html
