package org.hiddenmoon.waterreaction.sync;

import android.content.Context;

import androidx.annotation.NonNull;
import androidx.work.Worker;
import androidx.work.WorkerParameters;

import java.io.File;
import java.io.IOException;

public final class ResultUploadWorker extends Worker {
    public static final String INPUT_UPLOAD_ID = "upload_id";

    public ResultUploadWorker(
            @NonNull Context context,
            @NonNull WorkerParameters parameters) {
        super(context, parameters);
    }

    @NonNull
    @Override
    public Result doWork() {
        String uploadId = getInputData().getString(INPUT_UPLOAD_ID);
        if (uploadId == null || uploadId.isBlank()) return Result.failure();

        try (UploadQueue queue = new UploadQueue(getApplicationContext())) {
            UploadTask task = queue.claim(uploadId);
            if (task == null) return Result.success();
            UploadDecision decision;
            String responseCode;
            try {
                ServerConfig config = ServerConfig.fromAssets(getApplicationContext());
                UploadApi.Response response = UploadApi.create(
                        getApplicationContext(), config).upload(task.archive);
                decision = UploadDecision.forHttp(response.statusCode);
                responseCode = response.code;
            } catch (UploadApi.RegistrationException error) {
                decision = UploadDecision.forHttp(error.statusCode);
                responseCode = error.code;
            } catch (IOException error) {
                queue.retry(uploadId, 0, error.getMessage());
                UploadStateStore.record(getApplicationContext(),
                        UploadStateStore.NETWORK_ERROR, error.getMessage());
                return Result.retry();
            } catch (RuntimeException error) {
                queue.block(uploadId, error.getMessage());
                UploadStateStore.record(getApplicationContext(),
                        UploadStateStore.BLOCKED, error.getMessage());
                return Result.success();
            }
            return applyDecision(queue, task, decision, responseCode);
        }
    }

    private Result applyDecision(
            UploadQueue queue,
            UploadTask task,
            UploadDecision decision,
            String responseCode) {
        if (UploadCleanupPolicy.mayDeleteArchive(decision)) {
            boolean deleted = deletePendingArchive(task);
            if (UploadCleanupPolicy.mayDeleteQueueRow(decision, deleted)) {
                queue.deleteRow(task.uploadId);
                UploadStateStore.record(getApplicationContext(),
                        UploadStateStore.SUCCESS, responseCode);
                return Result.success();
            }
            queue.retry(task.uploadId, 0, "无法清理已上传结果");
            UploadStateStore.record(getApplicationContext(),
                    UploadStateStore.RETRY, "无法清理已上传结果");
            return Result.retry();
        }
        if (decision == UploadDecision.RETRY) {
            queue.retry(task.uploadId, 0, responseCode);
            UploadStateStore.record(getApplicationContext(), UploadStateStore.RETRY, responseCode);
            return Result.retry();
        }
        queue.block(task.uploadId, responseCode);
        UploadStateStore.record(getApplicationContext(), UploadStateStore.BLOCKED, responseCode);
        return Result.success();
    }

    private boolean deletePendingArchive(UploadTask task) {
        try {
            File root = new File(
                    getApplicationContext().getFilesDir(), "pending_uploads")
                    .getCanonicalFile();
            File archive = task.archive.getCanonicalFile();
            if (!root.equals(archive.getParentFile())
                    || !archive.getName().equals(task.uploadId + ".zip")) {
                return false;
            }
            return !archive.exists() || archive.delete();
        } catch (IOException error) {
            return false;
        }
    }
}
