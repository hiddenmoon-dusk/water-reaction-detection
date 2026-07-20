package org.hiddenmoon.waterreaction.sync;

import android.content.Context;
import android.content.SharedPreferences;

/** Stores only upload lifecycle metadata, never image or detection payload data. */
public final class UploadStateStore {
    public static final String SUCCESS = "success";
    public static final String RETRY = "retry";
    public static final String BLOCKED = "blocked";
    public static final String NETWORK_ERROR = "network_error";

    private static final String PREFERENCES = "upload-state";
    private static final String KEY_STATUS = "last_status";
    private static final String KEY_DETAIL = "last_detail";
    private static final String KEY_TIME = "last_time";

    private UploadStateStore() {}

    public static void record(Context context, String status, String detail) {
        SharedPreferences preferences = context.getApplicationContext()
                .getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE);
        preferences.edit()
                .putString(KEY_STATUS, status == null ? "unknown" : status)
                .putString(KEY_DETAIL, limit(detail))
                .putLong(KEY_TIME, System.currentTimeMillis())
                .apply();
    }

    public static Snapshot read(Context context) {
        SharedPreferences preferences = context.getApplicationContext()
                .getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE);
        String status = preferences.getString(KEY_STATUS, null);
        if (status == null) return null;
        return new Snapshot(
                status,
                preferences.getString(KEY_DETAIL, null),
                preferences.getLong(KEY_TIME, 0));
    }

    private static String limit(String value) {
        if (value == null) return null;
        return value.length() <= 200 ? value : value.substring(0, 200);
    }

    public static final class Snapshot {
        public final String status;
        public final String detail;
        public final long timeMillis;

        private Snapshot(String status, String detail, long timeMillis) {
            this.status = status;
            this.detail = detail;
            this.timeMillis = timeMillis;
        }
    }
}
