from __future__ import annotations

import pandas as pd

from roadrisk.models.explain import add_feature_display_labels, feature_display_label


def test_feature_display_label_decodes_condition_codes():
    assert feature_display_label("road_type_6") == "Road type: Single carriageway"
    assert feature_display_label("light_conditions_4") == "Light conditions: Darkness - lights lit"
    assert feature_display_label("weather_conditions_7") == "Weather: Fog or mist"


def test_feature_display_label_decodes_borough_and_boolean_features():
    assert feature_display_label("local_authority_ons_district_E09000001") == "Borough: City of London"
    assert feature_display_label("has_cyclist") == "Cyclist involved"


def test_add_feature_display_labels_preserves_original_feature_name():
    df = pd.DataFrame({"feature": ["road_surface_conditions_2"], "importance": [0.4]})
    labelled = add_feature_display_labels(df)
    assert labelled.loc[0, "feature"] == "road_surface_conditions_2"
    assert labelled.loc[0, "feature_label"] == "Road surface: Wet or damp"
