package org.hiddenmoon.waterreaction.sync;

import android.content.Context;

import androidx.work.BackoffPolicy;
import androidx.work.Constraints;
import androidx.work.Data;
import androidx.work.ExistingWorkPolicy;
import androidx.work.NetworkType;
import androidx.work.OneTimeWorkRequest;
import androidx.work.WorkManager;

import java.util.concurrent.TimeUnit;

public final class UploadScheduler {
    private UploadScheduler() {}

    public static String workName(String uploadId) {
        return "upload-" + uploadId;
    }

    public static void schedule(Context context, String uploadId) {
        Constraints constraints = new Constraints.Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                .build();
        Data input = new Data.Builder()
                .putString(ResultUploadWorker.INPUT_UPLOAD_ID, uploadId)
                .build();
        OneTimeWorkRequest request = new OneTimeWorkRequest.Builder(
                ResultUploadWorker.class)
                .setInputData(input)
                .setConstraints(constraints)
                .setBackoffCriteria(
                        BackoffPolicy.EXPONENTIAL, 10, TimeUnit.SECONDS)
                .build();
        WorkManager.getInstance(context).enqueueUniqueWork(
                workName(uploadId), ExistingWorkPolicy.KEEP, request);
    }

    public static void restorePending(Context context, UploadQueue queue) {
        for (String uploadId : queue.pendingIds()) {
            schedule(context, uploadId);
        }
    }
}
