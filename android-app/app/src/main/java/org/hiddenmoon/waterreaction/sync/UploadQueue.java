package org.hiddenmoon.waterreaction.sync;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public final class UploadQueue extends SQLiteOpenHelper {
    private static final String DATABASE_NAME = "upload-queue.db";
    private static final int DATABASE_VERSION = 1;
    public static final String PENDING = "pending";
    public static final String UPLOADING = "uploading";
    public static final String RETRY_WAIT = "retry_wait";
    public static final String BLOCKED = "blocked";

    public UploadQueue(Context context) {
        super(context.getApplicationContext(), DATABASE_NAME, null, DATABASE_VERSION);
        SQLiteDatabase database = getWritableDatabase();
        ContentValues reset = new ContentValues();
        reset.put("status", PENDING);
        reset.put("updated_at", System.currentTimeMillis());
        database.update("upload_tasks", reset, "status = ?", new String[]{UPLOADING});
    }

    @Override
    public void onCreate(SQLiteDatabase database) {
        database.execSQL("""
                CREATE TABLE upload_tasks (
                    upload_id TEXT PRIMARY KEY,
                    archive_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """);
        database.execSQL("CREATE INDEX upload_tasks_status ON upload_tasks(status, next_attempt_at)");
    }

    @Override
    public void onUpgrade(SQLiteDatabase database, int oldVersion, int newVersion) {
        throw new IllegalStateException("不支持上传队列数据库降级或未知升级");
    }

    public synchronized void enqueue(String uploadId, File archive) {
        long now = System.currentTimeMillis();
        ContentValues values = new ContentValues();
        values.put("upload_id", uploadId);
        values.put("archive_path", archive.getAbsolutePath());
        values.put("status", PENDING);
        values.put("attempts", 0);
        values.put("next_attempt_at", 0);
        values.putNull("last_error");
        values.put("created_at", now);
        values.put("updated_at", now);
        getWritableDatabase().insertWithOnConflict(
                "upload_tasks", null, values, SQLiteDatabase.CONFLICT_IGNORE);
    }

    public synchronized UploadTask get(String uploadId) {
        try (Cursor cursor = getReadableDatabase().query(
                "upload_tasks", null, "upload_id = ?", new String[]{uploadId},
                null, null, null, "1")) {
            return cursor.moveToFirst() ? fromCursor(cursor) : null;
        }
    }

    public synchronized UploadTask claim(String uploadId) {
        SQLiteDatabase database = getWritableDatabase();
        database.beginTransaction();
        try {
            UploadTask task = query(database, uploadId);
            long now = System.currentTimeMillis();
            if (task == null || BLOCKED.equals(task.status)
                    || RETRY_WAIT.equals(task.status) && task.nextAttemptAt > now) {
                database.setTransactionSuccessful();
                return null;
            }
            ContentValues values = new ContentValues();
            values.put("status", UPLOADING);
            values.put("attempts", task.attempts + 1);
            values.put("updated_at", now);
            values.putNull("last_error");
            database.update("upload_tasks", values, "upload_id = ?", new String[]{uploadId});
            database.setTransactionSuccessful();
            return query(database, uploadId);
        } finally {
            database.endTransaction();
        }
    }

    public synchronized void retry(String uploadId, long delayMillis, String error) {
        ContentValues values = new ContentValues();
        values.put("status", RETRY_WAIT);
        values.put("next_attempt_at", System.currentTimeMillis() + Math.max(0, delayMillis));
        values.put("last_error", limit(error));
        values.put("updated_at", System.currentTimeMillis());
        getWritableDatabase().update("upload_tasks", values, "upload_id = ?", new String[]{uploadId});
    }

    public synchronized void block(String uploadId, String error) {
        ContentValues values = new ContentValues();
        values.put("status", BLOCKED);
        values.put("next_attempt_at", 0);
        values.put("last_error", limit(error));
        values.put("updated_at", System.currentTimeMillis());
        getWritableDatabase().update("upload_tasks", values, "upload_id = ?", new String[]{uploadId});
    }

    public synchronized void deleteRow(String uploadId) {
        getWritableDatabase().delete("upload_tasks", "upload_id = ?", new String[]{uploadId});
    }

    public synchronized List<String> pendingIds() {
        List<String> ids = new ArrayList<>();
        try (Cursor cursor = getReadableDatabase().query(
                "upload_tasks", new String[]{"upload_id"},
                "status IN (?, ?)", new String[]{PENDING, RETRY_WAIT},
                null, null, "created_at, upload_id")) {
            while (cursor.moveToNext()) ids.add(cursor.getString(0));
        }
        return ids;
    }

    public synchronized UploadSummary summary() {
        int pending = 0;
        int uploading = 0;
        int retryWait = 0;
        int blocked = 0;
        try (Cursor cursor = getReadableDatabase().rawQuery(
                "SELECT status, COUNT(*) FROM upload_tasks GROUP BY status", null)) {
            while (cursor.moveToNext()) {
                String status = cursor.getString(0);
                int count = cursor.getInt(1);
                if (PENDING.equals(status)) pending = count;
                else if (UPLOADING.equals(status)) uploading = count;
                else if (RETRY_WAIT.equals(status)) retryWait = count;
                else if (BLOCKED.equals(status)) blocked = count;
            }
        }
        String latestError = null;
        try (Cursor cursor = getReadableDatabase().query(
                "upload_tasks", new String[]{"last_error"},
                "last_error IS NOT NULL AND last_error <> ''", null,
                null, null, "updated_at DESC", "1")) {
            if (cursor.moveToFirst()) latestError = cursor.getString(0);
        }
        return new UploadSummary(pending, uploading, retryWait, blocked, latestError);
    }

    /** Make retryable tasks immediately eligible for a manual retry. */
    public synchronized int releaseRetryWait() {
        ContentValues values = new ContentValues();
        values.put("status", PENDING);
        values.put("next_attempt_at", 0);
        values.put("updated_at", System.currentTimeMillis());
        return getWritableDatabase().update(
                "upload_tasks", values, "status = ?", new String[]{RETRY_WAIT});
    }

    /** Requeue one compatibility failure from a pre-fix Android archive. */
    public synchronized int releaseLegacyPayloadFailures() {
        ContentValues values = new ContentValues();
        values.put("status", PENDING);
        values.put("next_attempt_at", 0);
        values.putNull("last_error");
        values.put("updated_at", System.currentTimeMillis());
        return getWritableDatabase().update(
                "upload_tasks", values,
                "status = ? AND last_error = ? AND attempts <= ?",
                new String[]{BLOCKED, UploadCompatibilityPolicy.INVALID_PAYLOAD, "1"});
    }

    private static UploadTask query(SQLiteDatabase database, String uploadId) {
        try (Cursor cursor = database.query(
                "upload_tasks", null, "upload_id = ?", new String[]{uploadId},
                null, null, null, "1")) {
            return cursor.moveToFirst() ? fromCursor(cursor) : null;
        }
    }

    private static UploadTask fromCursor(Cursor cursor) {
        return new UploadTask(
                cursor.getString(cursor.getColumnIndexOrThrow("upload_id")),
                new File(cursor.getString(cursor.getColumnIndexOrThrow("archive_path"))),
                cursor.getString(cursor.getColumnIndexOrThrow("status")),
                cursor.getInt(cursor.getColumnIndexOrThrow("attempts")),
                cursor.getLong(cursor.getColumnIndexOrThrow("next_attempt_at")),
                cursor.isNull(cursor.getColumnIndexOrThrow("last_error"))
                        ? null : cursor.getString(cursor.getColumnIndexOrThrow("last_error")));
    }

    private static String limit(String value) {
        if (value == null) return "unknown";
        return value.length() <= 500 ? value : value.substring(0, 500);
    }
}
