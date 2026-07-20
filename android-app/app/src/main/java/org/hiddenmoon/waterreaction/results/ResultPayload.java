package org.hiddenmoon.waterreaction.results;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Set;

public final class ResultPayload {
    private static final Set<String> WATER_TYPES = Set.of("污水", "生活用水", "养殖水体");
    private static final Set<String> MODES = Set.of("normal", "scan", "manual");
    private static final Set<String> LABELS = Set.of("已反应", "未反应");

    private ResultPayload() {}

    public static String create(
            String uploadId,
            String waterType,
            String mode,
            String appReleaseId,
            int modelGeneration,
            int datasetGeneration,
            int appVersionCode,
            String deviceModel,
            List<Tube> tubes) {
        if (uploadId == null || uploadId.isBlank()) throw new IllegalArgumentException("uploadId 不能为空");
        if (!WATER_TYPES.contains(waterType)) throw new IllegalArgumentException("不支持的水样类型");
        if (!MODES.contains(mode)) throw new IllegalArgumentException("不支持的检测模式");
        if (appReleaseId == null || appReleaseId.isBlank()) throw new IllegalArgumentException("发布 ID 不能为空");
        if (modelGeneration <= 0 || datasetGeneration <= 0 || appVersionCode <= 0) {
            throw new IllegalArgumentException("版本和代次必须为正数");
        }
        if (tubes == null || tubes.isEmpty()) throw new IllegalArgumentException("没有检测结果");

        try {
            JSONArray results = new JSONArray();
            int id = 1;
            for (Tube tube : tubes) {
                tube.validate();
                results.put(new JSONObject()
                        .put("id", id++)
                        .put("x1", tube.x1)
                        .put("y1", tube.y1)
                        .put("x2", tube.x2)
                        .put("y2", tube.y2)
                        .put("label", tube.label)
                        .put("confidence", Math.round(tube.confidence * 10_000.0) / 10_000.0));
            }
            return new JSONObject()
                    .put("schema_version", 1)
                    .put("upload_id", uploadId)
                    .put("captured_at", capturedAtUtc())
                    .put("water_type", waterType)
                    .put("mode", mode)
                    .put("app_release_id", appReleaseId)
                    .put("model_generation", modelGeneration)
                    .put("dataset_generation", datasetGeneration)
                    .put("client_platform", "android")
                    .put("app_version_code", appVersionCode)
                    .put("device_model", deviceModel == null ? "" : deviceModel)
                    .put("results", results)
                    .toString();
        } catch (JSONException error) {
            throw new IllegalArgumentException("无法生成结果 JSON", error);
        }
    }

    /** Match the desktop client's Python datetime.isoformat() output. */
    private static String capturedAtUtc() {
        return LocalDateTime.now(ZoneOffset.UTC)
                .format(DateTimeFormatter.ISO_LOCAL_DATE_TIME) + "+00:00";
    }

    public static final class Tube {
        public final int x1;
        public final int y1;
        public final int x2;
        public final int y2;
        public final String label;
        public final float confidence;

        public Tube(int x1, int y1, int x2, int y2, String label, float confidence) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.label = label;
            this.confidence = confidence;
        }

        private void validate() {
            if (x1 < 0 || y1 < 0 || x2 <= x1 || y2 <= y1) {
                throw new IllegalArgumentException("检测框坐标无效");
            }
            if (!LABELS.contains(label)) throw new IllegalArgumentException("检测标签无效");
            if (!Float.isFinite(confidence) || confidence < 0 || confidence > 1) {
                throw new IllegalArgumentException("检测置信度无效");
            }
        }
    }
}
