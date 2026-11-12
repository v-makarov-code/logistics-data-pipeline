-- The input is the FULL test + future snapshot for exactly one model and as_of.
DELETE FROM mart.ml_price_predictions target
WHERE target.model_version = %(model_version)s AND target.as_of_date = %(as_of_date)s
  AND NOT EXISTS (
      SELECT 1 FROM prediction_stage stage
      WHERE stage.model_version = target.model_version
        AND stage.as_of_date = target.as_of_date
        AND stage.source_price_id = target.source_price_id
  );
