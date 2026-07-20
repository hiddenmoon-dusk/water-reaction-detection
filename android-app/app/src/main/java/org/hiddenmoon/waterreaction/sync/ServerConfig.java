package org.hiddenmoon.waterreaction.sync;

import android.content.Context;

import org.json.JSONObject;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;

public final class ServerConfig {
    public final String apiBaseUrl;
    public final String bootstrapToken;
    public final String appReleaseId;
    public final int modelGeneration;
    public final int datasetGeneration;

    public ServerConfig(
            String apiBaseUrl,
            String bootstrapToken,
            String appReleaseId,
            int modelGeneration,
            int datasetGeneration) {
        this.apiBaseUrl = require(apiBaseUrl, "服务器地址").replaceAll("/+$", "");
        this.bootstrapToken = require(bootstrapToken, "bootstrap token");
        this.appReleaseId = require(appReleaseId, "发布 ID");
        if (!this.apiBaseUrl.startsWith("http://") && !this.apiBaseUrl.startsWith("https://")) {
            throw new IllegalArgumentException("服务器地址必须是 HTTP(S)");
        }
        if (modelGeneration <= 0 || datasetGeneration <= 0) {
            throw new IllegalArgumentException("服务器代次必须为正数");
        }
        this.modelGeneration = modelGeneration;
        this.datasetGeneration = datasetGeneration;
    }

    public static ServerConfig fromAssets(Context context) {
        try (InputStream input = context.getAssets().open("server-config.json")) {
            JSONObject json = new JSONObject(new String(input.readAllBytes(), StandardCharsets.UTF_8));
            return new ServerConfig(
                    json.getString("api_base_url"),
                    json.getString("bootstrap_token"),
                    json.getString("app_release_id"),
                    json.getInt("model_generation"),
                    json.getInt("dataset_generation"));
        } catch (Exception error) {
            throw new IllegalStateException("服务器配置无效", error);
        }
    }

    private static String require(String value, String name) {
        if (value == null || value.isBlank()) throw new IllegalArgumentException(name + "不能为空");
        return value.trim();
    }
}
