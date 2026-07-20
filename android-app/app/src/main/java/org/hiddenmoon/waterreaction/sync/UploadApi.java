package org.hiddenmoon.waterreaction.sync;

import android.content.Context;
import android.content.SharedPreferences;

import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.time.Duration;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;

public final class UploadApi {
    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");
    private static final MediaType ZIP = MediaType.get("application/zip");
    private final ServerConfig config;
    private final CredentialStore credentialStore;
    private final OkHttpClient client;

    public UploadApi(ServerConfig config, CredentialStore credentialStore, OkHttpClient client) {
        this.config = config;
        this.credentialStore = credentialStore;
        this.client = client == null ? new OkHttpClient.Builder()
                .connectTimeout(Duration.ofSeconds(10))
                .readTimeout(Duration.ofSeconds(90))
                .writeTimeout(Duration.ofSeconds(90))
                .build() : client;
    }

    public static UploadApi create(Context context, ServerConfig config) {
        return new UploadApi(config, new SharedPreferenceCredentials(context), null);
    }

    public Response upload(File archive) throws IOException {
        Credentials credentials = credentialStore.load();
        if (credentials == null) credentials = register();
        Response response = uploadOnce(archive, credentials);
        if (response.statusCode == 401 && ("invalid_credentials".equals(response.code)
                || "authentication_required".equals(response.code))) {
            credentialStore.clear();
            response = uploadOnce(archive, register());
        }
        return response;
    }

    private Credentials register() throws IOException {
        try {
            String json = new JSONObject()
                    .put("bootstrap_token", config.bootstrapToken)
                    .put("app_release_id", config.appReleaseId)
                    .put("model_generation", config.modelGeneration)
                    .put("client_platform", "android")
                    .toString();
            Request request = new Request.Builder()
                    .url(config.apiBaseUrl + "/api/v1/client/register")
                    .post(RequestBody.create(json, JSON))
                    .build();
            try (okhttp3.Response response = client.newCall(request).execute()) {
                String text = response.body() == null ? "" : response.body().string();
                if (response.code() < 200 || response.code() >= 300) {
                    throw new RegistrationException(response.code(), responseCode(text, response.code()));
                }
                JSONObject payload = new JSONObject(text);
                Credentials credentials = new Credentials(
                        payload.getString("installation_id"), payload.getString("token"));
                credentialStore.save(credentials);
                return credentials;
            }
        } catch (RegistrationException error) {
            throw error;
        } catch (IOException error) {
            throw error;
        } catch (Exception error) {
            throw new IOException("服务器注册响应无效", error);
        }
    }

    private Response uploadOnce(File archive, Credentials credentials) throws IOException {
        RequestBody multipart = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", archive.getName(), RequestBody.create(archive, ZIP))
                .build();
        Request request = new Request.Builder()
                .url(config.apiBaseUrl + "/api/v1/results")
                .header("Authorization", "Bearer " + credentials.token)
                .header("X-Installation-ID", credentials.installationId)
                .post(multipart)
                .build();
        try (okhttp3.Response response = client.newCall(request).execute()) {
            String text = response.body() == null ? "" : response.body().string();
            return new Response(response.code(), responseCode(text, response.code()));
        }
    }

    private static String responseCode(String text, int statusCode) {
        try {
            JSONObject json = new JSONObject(text);
            String code = json.optString("code");
            if (code.isBlank()) code = json.optString("status");
            return code.isBlank() ? "http_" + statusCode : code;
        } catch (Exception error) {
            return "http_" + statusCode;
        }
    }

    public interface CredentialStore {
        Credentials load();
        void save(Credentials credentials);
        void clear();
    }

    public static final class Credentials {
        public final String installationId;
        public final String token;
        public Credentials(String installationId, String token) {
            this.installationId = installationId;
            this.token = token;
        }
    }

    public static final class Response {
        public final int statusCode;
        public final String code;
        Response(int statusCode, String code) {
            this.statusCode = statusCode;
            this.code = code;
        }
    }

    public static final class RegistrationException extends IOException {
        public final int statusCode;
        public final String code;
        RegistrationException(int statusCode, String code) {
            super("registration failed: " + code);
            this.statusCode = statusCode;
            this.code = code;
        }
    }

    private static final class SharedPreferenceCredentials implements CredentialStore {
        private final SharedPreferences preferences;
        SharedPreferenceCredentials(Context context) {
            preferences = context.getSharedPreferences("server-credentials", Context.MODE_PRIVATE);
        }
        @Override public Credentials load() {
            String installationId = preferences.getString("installation_id", null);
            String token = preferences.getString("token", null);
            return installationId == null || token == null ? null : new Credentials(installationId, token);
        }
        @Override public void save(Credentials credentials) {
            preferences.edit().putString("installation_id", credentials.installationId)
                    .putString("token", credentials.token).apply();
        }
        @Override public void clear() { preferences.edit().clear().apply(); }
    }
}
