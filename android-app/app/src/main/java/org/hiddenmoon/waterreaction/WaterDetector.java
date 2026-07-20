package org.hiddenmoon.waterreaction;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;

import org.tensorflow.lite.Interpreter;

import java.io.Closeable;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public final class WaterDetector implements Closeable {
    public interface ProgressListener {
        void onProgress(int completed, int total);
    }

    public static final class Result {
        public final RectF box;
        public final float detectionScore;
        public final float probability;
        public final String label;

        Result(RectF box, float detectionScore, float probability) {
            this.box = box;
            this.detectionScore = detectionScore;
            this.probability = probability;
            this.label = probability > 0.5f ? "已反应" : "未反应";
        }

        public float confidence() {
            return probability > 0.5f ? probability : 1f - probability;
        }
    }

    private static final int DETECTOR_SIZE = 640;
    private static final int CLASSIFIER_SIZE = 128;
    private static final float DETECTOR_CONFIDENCE = 0.3f;
    private static final float NMS_IOU = 0.45f;

    private final Interpreter detector;
    private final Interpreter classifier;
    private final ByteBuffer detectorInput = directFloatBuffer(DETECTOR_SIZE * DETECTOR_SIZE * 3);
    private final ByteBuffer classifierInput = directFloatBuffer(CLASSIFIER_SIZE * CLASSIFIER_SIZE * 3);

    public WaterDetector(Context context) throws IOException {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors() - 1)));
        detector = new Interpreter(mapAsset(context, "detector.tflite"), options);
        classifier = new Interpreter(mapAsset(context, "classifier.tflite"), options);
    }

    public List<Result> analyzeNormal(Bitmap bitmap) {
        return classify(bitmap, nms(detectTile(bitmap, 0, 0), NMS_IOU));
    }

    public List<Result> analyzeScan(Bitmap bitmap, ProgressListener listener) {
        int tile = 640;
        int step = 512;
        List<Integer> xs = origins(bitmap.getWidth(), tile, step);
        List<Integer> ys = origins(bitmap.getHeight(), tile, step);
        List<Candidate> candidates = new ArrayList<>();
        int total = xs.size() * ys.size();
        int completed = 0;
        for (int y : ys) {
            for (int x : xs) {
                int width = Math.min(tile, bitmap.getWidth() - x);
                int height = Math.min(tile, bitmap.getHeight() - y);
                Bitmap crop = Bitmap.createBitmap(bitmap, x, y, width, height);
                candidates.addAll(detectTile(crop, x, y));
                if (crop != bitmap) crop.recycle();
                completed++;
                if (listener != null) listener.onProgress(completed, total);
            }
        }
        return classify(bitmap, nms(candidates, NMS_IOU));
    }

    public List<Result> analyzeManual(Bitmap bitmap, List<RectF> boxes) {
        List<Result> results = new ArrayList<>();
        for (RectF requested : boxes) {
            RectF box = clipped(requested, bitmap.getWidth(), bitmap.getHeight());
            if (box.width() < 10 || box.height() < 10) continue;
            results.add(new Result(box, 1f, classifyCrop(bitmap, box)));
        }
        return results;
    }

    private List<Result> classify(Bitmap source, List<Candidate> candidates) {
        List<Result> results = new ArrayList<>();
        for (Candidate candidate : candidates) {
            RectF box = clipped(candidate.box, source.getWidth(), source.getHeight());
            if (box.width() < 2 || box.height() < 2) continue;
            results.add(new Result(box, candidate.score, classifyCrop(source, box)));
        }
        return results;
    }

    private float classifyCrop(Bitmap source, RectF box) {
        int left = Math.max(0, (int) Math.floor(box.left));
        int top = Math.max(0, (int) Math.floor(box.top));
        int right = Math.min(source.getWidth(), (int) Math.ceil(box.right));
        int bottom = Math.min(source.getHeight(), (int) Math.ceil(box.bottom));
        Bitmap crop = Bitmap.createBitmap(source, left, top, right - left, bottom - top);
        Bitmap resized = Bitmap.createScaledBitmap(crop, CLASSIFIER_SIZE, CLASSIFIER_SIZE, true);
        int[] pixels = new int[CLASSIFIER_SIZE * CLASSIFIER_SIZE];
        resized.getPixels(pixels, 0, CLASSIFIER_SIZE, 0, 0, CLASSIFIER_SIZE, CLASSIFIER_SIZE);
        classifierInput.rewind();
        for (int pixel : pixels) {
            classifierInput.putFloat(Color.red(pixel));
            classifierInput.putFloat(Color.green(pixel));
            classifierInput.putFloat(Color.blue(pixel));
        }
        classifierInput.rewind();
        float[][] output = new float[1][1];
        classifier.run(classifierInput, output);
        if (resized != crop) resized.recycle();
        crop.recycle();
        return Math.max(0f, Math.min(1f, output[0][0]));
    }

    private List<Candidate> detectTile(Bitmap source, int offsetX, int offsetY) {
        float scale = Math.min((float) DETECTOR_SIZE / source.getWidth(),
                (float) DETECTOR_SIZE / source.getHeight());
        int scaledWidth = Math.max(1, Math.round(source.getWidth() * scale));
        int scaledHeight = Math.max(1, Math.round(source.getHeight() * scale));
        float padX = (DETECTOR_SIZE - scaledWidth) / 2f;
        float padY = (DETECTOR_SIZE - scaledHeight) / 2f;

        Bitmap inputBitmap = Bitmap.createBitmap(DETECTOR_SIZE, DETECTOR_SIZE, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(inputBitmap);
        canvas.drawColor(Color.rgb(114, 114, 114));
        Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG);
        canvas.drawBitmap(source, null,
                new RectF(padX, padY, padX + scaledWidth, padY + scaledHeight), paint);

        int[] pixels = new int[DETECTOR_SIZE * DETECTOR_SIZE];
        inputBitmap.getPixels(pixels, 0, DETECTOR_SIZE, 0, 0, DETECTOR_SIZE, DETECTOR_SIZE);
        detectorInput.rewind();
        for (int pixel : pixels) {
            detectorInput.putFloat(Color.red(pixel) / 255f);
            detectorInput.putFloat(Color.green(pixel) / 255f);
            detectorInput.putFloat(Color.blue(pixel) / 255f);
        }
        detectorInput.rewind();
        float[][][] output = new float[1][5][8400];
        detector.run(detectorInput, output);
        inputBitmap.recycle();

        List<Candidate> candidates = new ArrayList<>();
        for (int i = 0; i < 8400; i++) {
            float score = output[0][4][i];
            if (score < DETECTOR_CONFIDENCE) continue;
            float cx = output[0][0][i] * DETECTOR_SIZE;
            float cy = output[0][1][i] * DETECTOR_SIZE;
            float width = output[0][2][i] * DETECTOR_SIZE;
            float height = output[0][3][i] * DETECTOR_SIZE;
            RectF box = new RectF(
                    (cx - width / 2f - padX) / scale + offsetX,
                    (cy - height / 2f - padY) / scale + offsetY,
                    (cx + width / 2f - padX) / scale + offsetX,
                    (cy + height / 2f - padY) / scale + offsetY);
            box = clipped(box, source.getWidth() + offsetX, source.getHeight() + offsetY);
            box.left = Math.max(offsetX, box.left);
            box.top = Math.max(offsetY, box.top);
            if (box.width() > 1 && box.height() > 1) candidates.add(new Candidate(box, score));
        }
        return candidates;
    }

    private static List<Integer> origins(int length, int tile, int step) {
        if (length <= tile) return Collections.singletonList(0);
        List<Integer> values = new ArrayList<>();
        for (int value = 0; value < length - tile; value += step) values.add(value);
        int edge = length - tile;
        if (values.isEmpty() || values.get(values.size() - 1) != edge) values.add(edge);
        return values;
    }

    private static List<Candidate> nms(List<Candidate> candidates, float threshold) {
        candidates.sort(Comparator.comparingDouble((Candidate item) -> item.score).reversed());
        List<Candidate> kept = new ArrayList<>();
        for (Candidate candidate : candidates) {
            boolean overlaps = false;
            for (Candidate previous : kept) {
                if (iou(candidate.box, previous.box) > threshold) {
                    overlaps = true;
                    break;
                }
            }
            if (!overlaps) kept.add(candidate);
            if (kept.size() >= 100) break;
        }
        return kept;
    }

    private static float iou(RectF a, RectF b) {
        float intersection = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left))
                * Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
        float union = a.width() * a.height() + b.width() * b.height() - intersection;
        return union <= 0 ? 0 : intersection / union;
    }

    private static RectF clipped(RectF box, int width, int height) {
        return new RectF(
                Math.max(0, Math.min(width, box.left)),
                Math.max(0, Math.min(height, box.top)),
                Math.max(0, Math.min(width, box.right)),
                Math.max(0, Math.min(height, box.bottom)));
    }

    private static ByteBuffer directFloatBuffer(int floatCount) {
        return ByteBuffer.allocateDirect(floatCount * 4).order(ByteOrder.nativeOrder());
    }

    private static MappedByteBuffer mapAsset(Context context, String name) throws IOException {
        try (AssetFileDescriptor descriptor = context.getAssets().openFd(name);
             FileInputStream input = new FileInputStream(descriptor.getFileDescriptor())) {
            return input.getChannel().map(FileChannel.MapMode.READ_ONLY,
                    descriptor.getStartOffset(), descriptor.getDeclaredLength());
        }
    }

    @Override
    public void close() {
        detector.close();
        classifier.close();
    }

    private static final class Candidate {
        final RectF box;
        final float score;

        Candidate(RectF box, float score) {
            this.box = box;
            this.score = score;
        }
    }
}
