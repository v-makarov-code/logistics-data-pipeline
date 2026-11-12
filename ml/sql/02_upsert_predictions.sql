INSERT INTO mart.ml_price_predictions AS target (
    source_price_id, model_version, as_of_date, dataset_split, loading_date,
    loading_city_gcid, unloading_city_gcid, loading_city_name, unloading_city_name,
    body_group, weight_category, volume_category, actual_price_rub, predicted_price_rub,
    model_name, model_run_id, dataset_run_id, scored_at, artifact_sha256
)
SELECT source_price_id, model_version, as_of_date, dataset_split, loading_date,
       loading_city_gcid, unloading_city_gcid, loading_city_name, unloading_city_name,
       body_group, weight_category, volume_category, actual_price_rub, predicted_price_rub,
       model_name, model_run_id, dataset_run_id, scored_at, artifact_sha256
FROM prediction_stage
ON CONFLICT (model_version, as_of_date, source_price_id) DO UPDATE SET
    dataset_split = EXCLUDED.dataset_split,
    loading_date = EXCLUDED.loading_date,
    loading_city_gcid = EXCLUDED.loading_city_gcid,
    unloading_city_gcid = EXCLUDED.unloading_city_gcid,
    loading_city_name = EXCLUDED.loading_city_name,
    unloading_city_name = EXCLUDED.unloading_city_name,
    body_group = EXCLUDED.body_group,
    weight_category = EXCLUDED.weight_category,
    volume_category = EXCLUDED.volume_category,
    actual_price_rub = EXCLUDED.actual_price_rub,
    predicted_price_rub = EXCLUDED.predicted_price_rub,
    model_name = EXCLUDED.model_name,
    model_run_id = EXCLUDED.model_run_id,
    dataset_run_id = EXCLUDED.dataset_run_id,
    scored_at = EXCLUDED.scored_at,
    artifact_sha256 = EXCLUDED.artifact_sha256,
    loaded_at = now()
-- A retry of the SAME artifact changes neither values nor loaded_at.
WHERE ROW(target.dataset_split, target.loading_date,
          target.loading_city_gcid, target.unloading_city_gcid,
          target.loading_city_name, target.unloading_city_name,
          target.body_group, target.weight_category, target.volume_category,
          target.actual_price_rub, target.predicted_price_rub,
          target.model_name, target.model_run_id, target.dataset_run_id,
          target.scored_at, target.artifact_sha256)
      IS DISTINCT FROM
      ROW(EXCLUDED.dataset_split, EXCLUDED.loading_date,
          EXCLUDED.loading_city_gcid, EXCLUDED.unloading_city_gcid,
          EXCLUDED.loading_city_name, EXCLUDED.unloading_city_name,
          EXCLUDED.body_group, EXCLUDED.weight_category, EXCLUDED.volume_category,
          EXCLUDED.actual_price_rub, EXCLUDED.predicted_price_rub,
          EXCLUDED.model_name, EXCLUDED.model_run_id, EXCLUDED.dataset_run_id,
          EXCLUDED.scored_at, EXCLUDED.artifact_sha256);
