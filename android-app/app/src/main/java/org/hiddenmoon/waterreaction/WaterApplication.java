package org.hiddenmoon.waterreaction;

import android.app.Application;

import org.hiddenmoon.waterreaction.capture.CaptureFiles;
import org.hiddenmoon.waterreaction.sync.UploadQueue;
import org.hiddenmoon.waterreaction.sync.UploadScheduler;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

public final class WaterApplication extends Application {
    @Override
    public void onCreate() {
        super.onCreate();
        try (UploadQueue queue = new UploadQueue(this)) {
            queue.releaseLegacyPayloadFailures();
            UploadScheduler.restorePending(this, queue);
        }
        try {
            File captureRoot = CaptureFiles.root(getCacheDir());
            long cutoff = System.currentTimeMillis() - TimeUnit.HOURS.toMillis(24);
            CaptureFiles.deleteStale(captureRoot, cutoff);
        } catch (IOException ignored) {
            // A later camera launch will recreate the private cache directory.
        }
    }
}
