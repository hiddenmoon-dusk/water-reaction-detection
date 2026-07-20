package org.hiddenmoon.waterreaction;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.ClipData;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.Gravity;
import android.view.View;
import android.view.animation.DecelerateInterpolator;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.core.content.FileProvider;

import org.hiddenmoon.waterreaction.capture.CaptureFiles;
import org.hiddenmoon.waterreaction.results.ResultArchive;
import org.hiddenmoon.waterreaction.results.ResultPayload;
import org.hiddenmoon.waterreaction.sync.ServerConfig;
import org.hiddenmoon.waterreaction.sync.UploadQueue;
import org.hiddenmoon.waterreaction.sync.UploadScheduler;
import org.hiddenmoon.waterreaction.sync.UploadStateStore;
import org.hiddenmoon.waterreaction.sync.UploadSummary;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

public final class MainActivity extends Activity {
    private static final int REQUEST_GALLERY = 100;
    private static final int REQUEST_CAMERA = 101;
    private static final int REQUEST_CAMERA_PERMISSION = 102;
    private static final String STATE_CAMERA_PATH = "camera_path";
    private static final String STATE_WATER_TYPE = "water_type";
    private static final String STATE_PHOTO_SOURCE = "photo_source";
    private static final String PHOTO_CAMERA = "camera";
    private static final String PHOTO_GALLERY = "gallery";

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final AtomicInteger analysisVersion = new AtomicInteger();
    private final CameraCaptureState cameraCaptureState = new CameraCaptureState();
    private WaterDetector detector;
    private Bitmap sourceBitmap;
    private File currentCaptureFile;
    private String photoSource;
    private String waterType;
    private String currentMode = "默认检测";
    private DetectionView detectionView;
    private TextView statusView;
    private List<WaterDetector.Result> currentResults = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setStatusBarColor(UiPalette.PAPER);
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR);
        if (savedInstanceState != null) {
            String cameraPath = savedInstanceState.getString(STATE_CAMERA_PATH);
            cameraCaptureState.restore(cameraPath);
            currentCaptureFile = cameraPath == null ? null : new File(cameraPath);
            waterType = savedInstanceState.getString(STATE_WATER_TYPE);
            photoSource = savedInstanceState.getString(STATE_PHOTO_SOURCE);
        }
        if (waterType == null) showWaterSelection();
        else showPhotoSource();
    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);
        outState.putString(STATE_CAMERA_PATH, cameraCaptureState.pendingUri());
        outState.putString(STATE_WATER_TYPE, waterType);
        outState.putString(STATE_PHOTO_SOURCE, photoSource);
    }

    private void showWaterSelection() {
        deleteCurrentCapture();
        waterType = null;
        LinearLayout root = verticalRoot();
        root.setGravity(Gravity.CENTER_HORIZONTAL);
        root.addView(headerBar("水体反应管检测系统"));
        root.addView(subtitle("请选择本次检测的水样类型"));
        TextView code = subtitle("ARCHIVE 01  /  WATER REACTION TUBE");
        code.setTextSize(11);
        code.setTextColor(UiPalette.SIGNAL_CYAN);
        root.addView(code);
        addWaterButton(root, "污水");
        addWaterButton(root, "生活用水");
        addWaterButton(root, "养殖水体");
        setContentView(root);
    }

    private void addWaterButton(LinearLayout root, String name) {
        Button button = primaryButton(name);
        button.setOnClickListener(view -> {
            waterType = name;
            showPhotoSource();
        });
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(72));
        params.setMargins(dp(20), dp(12), dp(20), 0);
        root.addView(button, params);
        enter(button, 40L);
    }

    private void showPhotoSource() {
        LinearLayout root = verticalRoot();
        root.setGravity(Gravity.CENTER_HORIZONTAL);
        root.addView(headerBar("选择照片来源"));
        root.addView(subtitle("当前水样：" + waterType));
        TextView code = subtitle("SOURCE SELECT  /  CAMERA OR ARCHIVE");
        code.setTextSize(11);
        code.setTextColor(UiPalette.SIGNAL_CYAN);
        root.addView(code);

        Button camera = primaryButton("拍照检测");
        camera.setOnClickListener(view -> startCamera());
        addLargeAction(root, camera);

        Button gallery = primaryButton("从相册选择");
        gallery.setOnClickListener(view -> {
            deleteCurrentCapture();
            Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            intent.setType("image/*");
            startActivityForResult(intent, REQUEST_GALLERY);
        });
        addLargeAction(root, gallery);

        Button back = secondaryButton("更改水样");
        back.setOnClickListener(view -> showWaterSelection());
        addLargeAction(root, back);
        setContentView(root);
    }

    private void addLargeAction(LinearLayout root, Button button) {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(68));
        params.setMargins(dp(20), dp(14), dp(20), 0);
        root.addView(button, params);
        enter(button, 80L);
    }

    private void startCamera() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
            return;
        }
        try {
            deleteCurrentCapture();
            currentCaptureFile = CaptureFiles.create(getCacheDir());
            photoSource = PHOTO_CAMERA;
            cameraCaptureState.begin(currentCaptureFile.getAbsolutePath());
            Uri captureUri = FileProvider.getUriForFile(
                    this, BuildConfig.APPLICATION_ID + ".files", currentCaptureFile);
            Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            intent.putExtra(MediaStore.EXTRA_OUTPUT, captureUri);
            intent.setClipData(ClipData.newRawUri("water-photo", captureUri));
            intent.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION | Intent.FLAG_GRANT_READ_URI_PERMISSION);
            startActivityForResult(intent, REQUEST_CAMERA);
        } catch (IOException | IllegalArgumentException error) {
            deleteCurrentCapture();
            toast("无法创建拍照文件");
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) startCamera();
            else toast("需要相机权限才能拍照，也可以从相册选择");
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        Uri uri = null;
        if (requestCode == REQUEST_GALLERY && resultCode == RESULT_OK && data != null) {
            photoSource = PHOTO_GALLERY;
            uri = data.getData();
        } else if (requestCode == REQUEST_CAMERA) {
            String pendingValue = cameraCaptureState.pendingUri();
            File pendingFile = pendingValue == null ? null : new File(pendingValue);
            File captureRoot = captureRootOrNull();
            boolean imageWasWritten = captureRoot != null
                    && CaptureFiles.isUsable(captureRoot, pendingFile);
            String acceptedValue = cameraCaptureState.consume(resultCode == RESULT_OK, imageWasWritten);
            if (acceptedValue != null && imageWasWritten) {
                currentCaptureFile = new File(acceptedValue);
                photoSource = PHOTO_CAMERA;
                loadCapturedImage(currentCaptureFile);
                return;
            }
            deleteCurrentCapture();
        }
        if (uri != null) loadImage(uri);
    }

    private void loadCapturedImage(File file) {
        loadImageSource(ImageDecoder.createSource(file));
    }

    private void loadImage(Uri uri) {
        loadImageSource(ImageDecoder.createSource(getContentResolver(), uri));
    }

    private void loadImageSource(ImageDecoder.Source source) {
        try {
            sourceBitmap = ImageDecoder.decodeBitmap(source, (decoder, info, ignored) -> {
                decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
                int max = Math.max(info.getSize().getWidth(), info.getSize().getHeight());
                if (max > 3000) decoder.setTargetSampleSize((int) Math.ceil(max / 3000.0));
            });
            showDetection();
            runAnalysis("默认检测");
        } catch (IOException | RuntimeException error) {
            toast("照片读取失败，请换一张照片");
        }
    }

    private void showDetection() {
        LinearLayout root = verticalRoot();
        root.addView(headerBar("反应管检测\n" + waterType + " · " + currentMode));

        detectionView = new DetectionView(this);
        detectionView.setBitmap(sourceBitmap);
        root.addView(detectionView, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1));

        statusView = new TextView(this);
        statusView.setText("正在准备检测…");
        statusView.setTextSize(16);
        statusView.setTextColor(UiPalette.INK);
        statusView.setBackground(surface(UiPalette.MIST, UiPalette.GRID, 1));
        statusView.setPadding(dp(16), dp(8), dp(16), dp(8));
        root.addView(statusView);
        enter(statusView, 80L);

        LinearLayout modes = horizontalRow();
        Button normal = secondaryButton("默认检测");
        normal.setOnClickListener(view -> runAnalysis("默认检测"));
        modes.addView(normal, weighted());
        Button scan = secondaryButton("扫描检测");
        scan.setOnClickListener(view -> runAnalysis("扫描检测"));
        modes.addView(scan, weighted());
        Button manual = secondaryButton("手动框选");
        manual.setOnClickListener(view -> {
            if (!detectionView.isManualMode()) {
                analysisVersion.incrementAndGet();
                currentMode = "手动框选";
                detectionView.setManualMode(true);
                statusView.setText("请在图片上拖动框选；完成后再点一次“手动框选”");
            } else if (detectionView.getManualBoxes().isEmpty()) {
                toast("请先拖动框选反应管");
            } else {
                runAnalysis("手动框选");
            }
        });
        modes.addView(manual, weighted());
        root.addView(modes);
        enter(modes, 120L);

        LinearLayout actions = horizontalRow();
        Button reselect = secondaryButton("重新选图");
        reselect.setOnClickListener(view -> {
            analysisVersion.incrementAndGet();
            deleteCurrentCapture();
            showPhotoSource();
        });
        actions.addView(reselect, weighted());
        Button save = primaryButton("保存结果");
        save.setOnClickListener(view -> saveResult());
        actions.addView(save, weighted());
        root.addView(actions);
        enter(actions, 160L);
        setContentView(root);
    }

    private void runAnalysis(String mode) {
        if (sourceBitmap == null || detectionView == null) return;
        currentMode = mode;
        int version = analysisVersion.incrementAndGet();
        List<android.graphics.RectF> manualBoxes = detectionView.getManualBoxes();
        detectionView.setManualMode(false);
        statusView.setText(mode.equals("扫描检测") ? "扫描检测 0%" : "正在检测，请稍候…");
        if (mode.equals("扫描检测") || mode.equals("默认检测")) {
            detectionView.startScanMotion();
        }
        executor.execute(() -> {
            try {
                if (detector == null) detector = new WaterDetector(getApplicationContext());
                List<WaterDetector.Result> results;
                if (mode.equals("扫描检测")) {
                    results = detector.analyzeScan(sourceBitmap, (completed, total) -> runOnUiThread(() -> {
                        if (version == analysisVersion.get())
                            statusView.setText("扫描检测 " + (completed * 100 / total) + "%");
                    }));
                } else if (mode.equals("手动框选")) {
                    results = detector.analyzeManual(sourceBitmap, manualBoxes);
                } else {
                    results = detector.analyzeNormal(sourceBitmap);
                }
                runOnUiThread(() -> {
                    if (version != analysisVersion.get()) return;
                    currentResults = results;
                    detectionView.setResults(results);
                    detectionView.showResultsMotion();
                    long reacted = results.stream().filter(item -> item.probability > 0.5f).count();
                    statusView.setText("检测完成：共 " + results.size() + " 管，已反应 " + reacted
                            + " 管，未反应 " + (results.size() - reacted) + " 管");
                });
            } catch (Exception error) {
                runOnUiThread(() -> {
                    if (detectionView != null) detectionView.stopMotion();
                    if (version == analysisVersion.get()) statusView.setText("检测失败：" + safeMessage(error));
                });
            }
        });
    }

    private void saveResult() {
        if (detectionView == null || currentResults.isEmpty()) {
            toast("请先完成检测");
            return;
        }
        Bitmap rendered = detectionView.renderAnnotated();
        if (rendered == null) return;
        Bitmap original = sourceBitmap;
        List<WaterDetector.Result> resultSnapshot = new ArrayList<>(currentResults);
        String sourceSnapshot = photoSource;
        String waterSnapshot = waterType;
        String modeSnapshot = currentMode;
        statusView.setText("正在保存并加入上传队列…");
        executor.execute(() -> archiveAndQueue(
                original, rendered, resultSnapshot, sourceSnapshot,
                waterSnapshot, modeSnapshot));
    }

    private void archiveAndQueue(
            Bitmap original,
            Bitmap rendered,
            List<WaterDetector.Result> results,
            String source,
            String selectedWater,
            String selectedMode) {
        File archive = null;
        boolean queued = false;
        try {
            String uploadId = UUID.randomUUID().toString();
            ServerConfig config = ServerConfig.fromAssets(this);
            List<ResultPayload.Tube> tubes = new ArrayList<>();
            for (WaterDetector.Result result : results) {
                tubes.add(new ResultPayload.Tube(
                        Math.max(0, (int) Math.floor(result.box.left)),
                        Math.max(0, (int) Math.floor(result.box.top)),
                        Math.max(1, (int) Math.ceil(result.box.right)),
                        Math.max(1, (int) Math.ceil(result.box.bottom)),
                        result.label,
                        result.confidence()));
            }
            String json = ResultPayload.create(
                    uploadId,
                    selectedWater,
                    serverMode(selectedMode),
                    config.appReleaseId,
                    config.modelGeneration,
                    config.datasetGeneration,
                    BuildConfig.VERSION_CODE,
                    Build.MODEL,
                    tubes);
            archive = ResultArchive.writeAtomically(
                    new File(getFilesDir(), "pending_uploads"),
                    uploadId,
                    compress(original, Bitmap.CompressFormat.JPEG, 95),
                    compress(rendered, Bitmap.CompressFormat.PNG, 100),
                    json.getBytes(StandardCharsets.UTF_8));
            try (UploadQueue queue = new UploadQueue(this)) {
                queue.enqueue(uploadId, archive);
            }
            queued = true;
            try {
                UploadScheduler.schedule(this, uploadId);
            } catch (RuntimeException ignored) {
                // Startup recovery will schedule the durable queue row.
            }
            runOnUiThread(() -> finishQueuedSave(source));
        } catch (Exception error) {
            if (!queued && archive != null) archive.delete();
            runOnUiThread(() -> {
                if (statusView != null) statusView.setText("保存失败，请重试");
                toast("保存失败：" + safeMessage(error));
            });
        } finally {
            rendered.recycle();
        }
    }

    private void finishQueuedSave(String source) {
        boolean deleteCapture = SourceCleanupPolicy.shouldDeleteCapture(source, true);
        if (deleteCapture) deleteCurrentCapture();
        if (sourceBitmap != null && !sourceBitmap.isRecycled()) sourceBitmap.recycle();
        sourceBitmap = null;
        detectionView = null;
        statusView = null;
        currentResults = new ArrayList<>();
        photoSource = null;
        toast("结果已加入上传队列；断网时将在下次打开 App 后自动上传");
        showPhotoSource();
    }

    private static byte[] compress(
            Bitmap bitmap, Bitmap.CompressFormat format, int quality)
            throws IOException {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        if (bitmap == null || !bitmap.compress(format, quality, output)) {
            throw new IOException("无法编码检测图片");
        }
        return output.toByteArray();
    }

    private static String serverMode(String mode) {
        if ("扫描检测".equals(mode)) return "scan";
        if ("手动框选".equals(mode)) return "manual";
        return "normal";
    }

    private LinearLayout verticalRoot() {
        LinearLayout root = new ResearchRootLayout(this);
        root.setPadding(0, dp(12), 0, dp(12));
        return root;
    }

    private LinearLayout horizontalRow() {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setPadding(dp(8), dp(4), dp(8), dp(4));
        return row;
    }

    private LinearLayout.LayoutParams weighted() {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(0, dp(54), 1);
        params.setMargins(dp(4), 0, dp(4), 0);
        return params;
    }

    private TextView title(String value) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextSize(28);
        view.setTextColor(UiPalette.INK);
        view.setGravity(Gravity.CENTER);
        view.setPadding(dp(16), dp(44), dp(16), dp(12));
        return view;
    }

    private GradientDrawable surface(int fill, int stroke, int strokeWidth) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(fill);
        drawable.setStroke(dp(strokeWidth), stroke);
        drawable.setCornerRadius(0f);
        return drawable;
    }

    private void enter(View view, long delayMs) {
        if (!android.animation.ValueAnimator.areAnimatorsEnabled()) {
            view.setAlpha(1f);
            return;
        }
        view.setAlpha(0f);
        view.setTranslationY(dp(8));
        view.animate().alpha(1f).translationY(0f).setStartDelay(delayMs)
                .setDuration(220L).setInterpolator(new DecelerateInterpolator()).start();
    }

    private LinearLayout headerBar(String value) {
        LinearLayout bar = new LinearLayout(this);
        bar.setOrientation(LinearLayout.HORIZONTAL);
        bar.setGravity(Gravity.CENTER_VERTICAL);
        bar.setPadding(dp(16), dp(8), dp(8), dp(8));
        bar.setBackground(surface(UiPalette.PAPER, UiPalette.INK, 1));

        TextView label = new TextView(this);
        label.setText("FIELD SAMPLE  /  " + value);
        label.setTextSize(18);
        label.setTextColor(UiPalette.INK);
        label.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        label.setGravity(Gravity.CENTER_VERTICAL);
        bar.addView(label, new LinearLayout.LayoutParams(0, dp(52), 1));

        Button upload = secondaryButton("SYNC");
        upload.setTextSize(12);
        upload.setTextColor(UiPalette.INK);
        upload.setBackground(surface(UiPalette.SIGNAL_YELLOW, UiPalette.INK, 1));
        upload.setOnClickListener(view -> showUploadStatusDialog());
        bar.addView(upload, new LinearLayout.LayoutParams(dp(84), dp(48)));
        enter(bar, 0L);
        return bar;
    }

    private void showUploadStatusDialog() {
        executor.execute(() -> {
            UploadSummary summary;
            try (UploadQueue queue = new UploadQueue(this)) {
                summary = queue.summary();
            }
            UploadStateStore.Snapshot last = UploadStateStore.read(this);
            UploadSummary snapshot = summary;
            runOnUiThread(() -> {
                if (isFinishing()) return;
                AlertDialog.Builder builder = new AlertDialog.Builder(this)
                        .setTitle("SYNC / 结果上传进度")
                        .setMessage(uploadSummaryText(snapshot, last))
                        .setPositiveButton("关闭", null);
                if (snapshot.canRetry()) {
                    builder.setNeutralButton("立即重试", (dialog, which) -> retryUploads());
                }
                builder.show();
            });
        });
    }

    private String uploadSummaryText(
            UploadSummary summary,
            UploadStateStore.Snapshot last) {
        StringBuilder text = new StringBuilder();
        text.append("待上传：").append(summary.pending).append('\n')
                .append("上传中：").append(summary.uploading).append('\n')
                .append("等待重试：").append(summary.retryWait).append('\n')
                .append("服务器拒绝：").append(summary.blocked);
        if (summary.latestError != null && !summary.latestError.isBlank()) {
            text.append("\n最近错误：").append(summary.latestError);
        }
        if (last != null) {
            text.append("\n最近状态：").append(uploadStatusLabel(last.status));
            if (last.detail != null && !last.detail.isBlank()) {
                text.append("（").append(last.detail).append("）");
            }
        }
        if (summary.total() == 0 && last == null) text.append("\n暂无上传记录");
        return text.toString();
    }

    private String uploadStatusLabel(String status) {
        if (UploadStateStore.SUCCESS.equals(status)) return "上传成功";
        if (UploadStateStore.RETRY.equals(status)) return "等待重试";
        if (UploadStateStore.BLOCKED.equals(status)) return "服务器拒绝";
        if (UploadStateStore.NETWORK_ERROR.equals(status)) return "网络异常";
        return status;
    }

    private void retryUploads() {
        executor.execute(() -> {
            try (UploadQueue queue = new UploadQueue(this)) {
                queue.releaseRetryWait();
                UploadScheduler.restorePending(this, queue);
            } catch (RuntimeException error) {
                runOnUiThread(() -> toast("暂时无法重试：" + safeMessage(error)));
                return;
            }
            runOnUiThread(() -> toast("已安排重试，上传将在后台继续"));
        });
    }

    private TextView subtitle(String value) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextSize(17);
        view.setTextColor(UiPalette.INK);
        view.setGravity(Gravity.CENTER);
        view.setPadding(dp(16), dp(8), dp(16), dp(18));
        return view;
    }

    private Button primaryButton(String text) {
        Button button = new Button(this);
        button.setText(text);
        button.setTextSize(16);
        button.setTextColor(UiPalette.INK);
        button.setAllCaps(false);
        button.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        button.setBackground(surface(UiPalette.SIGNAL_YELLOW, UiPalette.INK, 1));
        button.setMinHeight(0);
        button.setPadding(dp(12), 0, dp(12), 0);
        return button;
    }

    private Button secondaryButton(String text) {
        Button button = new Button(this);
        button.setText(text);
        button.setTextSize(14);
        button.setTextColor(UiPalette.INK);
        button.setAllCaps(false);
        button.setBackground(surface(UiPalette.PAPER, UiPalette.INK, 1));
        button.setMinHeight(0);
        button.setPadding(dp(12), 0, dp(12), 0);
        return button;
    }

    private void toast(String message) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show();
    }

    private String safeMessage(Exception error) {
        String message = error.getMessage();
        return message == null || message.isBlank() ? "模型运行异常" : message;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private File captureRootOrNull() {
        try {
            return CaptureFiles.root(getCacheDir());
        } catch (IOException error) {
            return null;
        }
    }

    private void deleteCurrentCapture() {
        File root = captureRootOrNull();
        if (PHOTO_CAMERA.equals(photoSource) && root != null && currentCaptureFile != null) {
            CaptureFiles.deleteOwned(root, currentCaptureFile);
        }
        currentCaptureFile = null;
        cameraCaptureState.restore(null);
        if (PHOTO_CAMERA.equals(photoSource)) photoSource = null;
    }

    @Override
    protected void onDestroy() {
        analysisVersion.incrementAndGet();
        executor.shutdownNow();
        if (detector != null) detector.close();
        if (isFinishing()) deleteCurrentCapture();
        super.onDestroy();
    }
}
