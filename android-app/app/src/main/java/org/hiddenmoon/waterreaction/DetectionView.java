package org.hiddenmoon.waterreaction;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.MotionEvent;
import android.view.View;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

public final class DetectionView extends View {
    private final Paint imagePaint = new Paint(Paint.FILTER_BITMAP_FLAG);
    private final Paint boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint gridPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint scanPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private Bitmap bitmap;
    private List<WaterDetector.Result> results = Collections.emptyList();
    private final List<RectF> manualBoxes = new ArrayList<>();
    private boolean manualMode;
    private RectF imageBounds = new RectF();
    private float startX;
    private float startY;
    private RectF preview;
    private float scanProgress = -1f;
    private float resultProgress = 1f;
    private android.animation.ValueAnimator motionAnimator;

    public DetectionView(Context context) {
        super(context);
        setBackgroundColor(UiPalette.MIST);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(dp(3));
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(dp(14));
        textPaint.setStyle(Paint.Style.FILL);
        gridPaint.setColor(0x6699A39E);
        gridPaint.setStrokeWidth(dp(1));
        scanPaint.setColor(UiPalette.SIGNAL_CYAN);
        scanPaint.setStrokeWidth(dp(2));
    }

    public void setBitmap(Bitmap bitmap) {
        this.bitmap = bitmap;
        manualBoxes.clear();
        results = Collections.emptyList();
        invalidate();
    }

    public void setResults(List<WaterDetector.Result> results) {
        this.results = new ArrayList<>(results);
        invalidate();
    }

    public void startScanMotion() {
        stopMotion();
        if (!android.animation.ValueAnimator.areAnimatorsEnabled()) {
            scanProgress = 1f;
            invalidate();
            return;
        }
        motionAnimator = android.animation.ValueAnimator.ofFloat(0f, 1f);
        motionAnimator.setDuration(820L);
        motionAnimator.addUpdateListener(animation -> {
            scanProgress = (Float) animation.getAnimatedValue();
            invalidate();
        });
        motionAnimator.addListener(new android.animation.AnimatorListenerAdapter() {
            @Override
            public void onAnimationEnd(android.animation.Animator animation) {
                scanProgress = -1f;
                invalidate();
            }
        });
        motionAnimator.start();
    }

    public void showResultsMotion() {
        stopMotion();
        resultProgress = 0f;
        if (!android.animation.ValueAnimator.areAnimatorsEnabled()) {
            resultProgress = 1f;
            invalidate();
            return;
        }
        motionAnimator = android.animation.ValueAnimator.ofFloat(0f, 1f);
        motionAnimator.setDuration(240L);
        motionAnimator.addUpdateListener(animation -> {
            resultProgress = (Float) animation.getAnimatedValue();
            invalidate();
        });
        motionAnimator.start();
    }

    void stopMotion() {
        if (motionAnimator != null) motionAnimator.cancel();
        motionAnimator = null;
        scanProgress = -1f;
        resultProgress = 1f;
        invalidate();
    }

    @Override
    protected void onDetachedFromWindow() {
        stopMotion();
        super.onDetachedFromWindow();
    }

    public void setManualMode(boolean enabled) {
        manualMode = enabled;
        if (enabled) results = Collections.emptyList();
        invalidate();
    }

    public boolean isManualMode() {
        return manualMode;
    }

    public List<RectF> getManualBoxes() {
        return new ArrayList<>(manualBoxes);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (bitmap == null) return;
        imageBounds = fitBounds(bitmap.getWidth(), bitmap.getHeight(), getWidth(), getHeight());
        canvas.drawBitmap(bitmap, null, imageBounds, imagePaint);
        for (float x = imageBounds.left; x < imageBounds.right; x += dp(32)) {
            canvas.drawLine(x, imageBounds.top, x, imageBounds.bottom, gridPaint);
        }
        for (float y = imageBounds.top; y < imageBounds.bottom; y += dp(32)) {
            canvas.drawLine(imageBounds.left, y, imageBounds.right, y, gridPaint);
        }
        if (scanProgress >= 0f && scanProgress <= 1f) {
            float y = imageBounds.top + imageBounds.height() * scanProgress;
            canvas.drawLine(imageBounds.left, y, imageBounds.right, y, scanPaint);
        }
        for (WaterDetector.Result result : results) drawResult(canvas, result);
        boxPaint.setColor(Color.YELLOW);
        for (RectF box : manualBoxes) canvas.drawRect(toView(box), boxPaint);
        if (preview != null) canvas.drawRect(toView(preview), boxPaint);
    }

    private void drawResult(Canvas canvas, WaterDetector.Result result) {
        boxPaint.setColor(result.probability > 0.5f ? Color.rgb(35, 210, 135) : Color.rgb(255, 170, 40));
        int oldBoxAlpha = boxPaint.getAlpha();
        int oldTextAlpha = textPaint.getAlpha();
        int alpha = Math.round(255f * resultProgress);
        boxPaint.setAlpha(alpha);
        textPaint.setAlpha(alpha);
        RectF viewBox = toView(result.box);
        canvas.drawRect(viewBox, boxPaint);
        String label = result.label + " " + String.format(Locale.CHINA, "%.0f%%", result.confidence() * 100);
        float width = textPaint.measureText(label);
        Paint background = new Paint(Paint.ANTI_ALIAS_FLAG);
        background.setColor((boxPaint.getColor() & 0x00FFFFFF) | (alpha << 24));
        canvas.drawRect(viewBox.left, Math.max(0, viewBox.top - dp(24)),
                viewBox.left + width + dp(12), viewBox.top, background);
        canvas.drawText(label, viewBox.left + dp(6), viewBox.top - dp(6), textPaint);
        boxPaint.setAlpha(oldBoxAlpha);
        textPaint.setAlpha(oldTextAlpha);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (!manualMode || bitmap == null) return false;
        float x = toImageX(event.getX());
        float y = toImageY(event.getY());
        switch (event.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
                startX = x;
                startY = y;
                preview = new RectF(x, y, x, y);
                invalidate();
                return true;
            case MotionEvent.ACTION_MOVE:
                preview = ordered(startX, startY, x, y);
                invalidate();
                return true;
            case MotionEvent.ACTION_UP:
                preview = ordered(startX, startY, x, y);
                if (preview.width() >= 10 && preview.height() >= 10) manualBoxes.add(preview);
                preview = null;
                invalidate();
                return true;
            default:
                return true;
        }
    }

    public Bitmap renderAnnotated() {
        if (bitmap == null) return null;
        Bitmap rendered = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(rendered);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(Math.max(3, bitmap.getWidth() / 300f));
        Paint labelPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        labelPaint.setTextSize(Math.max(28, bitmap.getWidth() / 35f));
        for (WaterDetector.Result result : results) {
            paint.setColor(result.probability > 0.5f ? Color.GREEN : Color.rgb(255, 165, 0));
            canvas.drawRect(result.box, paint);
            labelPaint.setColor(paint.getColor());
            canvas.drawText(result.label, result.box.left,
                    Math.max(labelPaint.getTextSize(), result.box.top), labelPaint);
        }
        return rendered;
    }

    private RectF toView(RectF box) {
        float sx = imageBounds.width() / bitmap.getWidth();
        float sy = imageBounds.height() / bitmap.getHeight();
        return new RectF(imageBounds.left + box.left * sx, imageBounds.top + box.top * sy,
                imageBounds.left + box.right * sx, imageBounds.top + box.bottom * sy);
    }

    private float toImageX(float x) {
        return clamp((x - imageBounds.left) * bitmap.getWidth() / imageBounds.width(), 0, bitmap.getWidth());
    }

    private float toImageY(float y) {
        return clamp((y - imageBounds.top) * bitmap.getHeight() / imageBounds.height(), 0, bitmap.getHeight());
    }

    private static RectF fitBounds(int sourceWidth, int sourceHeight, int width, int height) {
        float scale = Math.min((float) width / sourceWidth, (float) height / sourceHeight);
        float renderedWidth = sourceWidth * scale;
        float renderedHeight = sourceHeight * scale;
        float left = (width - renderedWidth) / 2f;
        float top = (height - renderedHeight) / 2f;
        return new RectF(left, top, left + renderedWidth, top + renderedHeight);
    }

    private static RectF ordered(float x1, float y1, float x2, float y2) {
        return new RectF(Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2));
    }

    private static float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    private float dp(float value) {
        return value * getResources().getDisplayMetrics().density;
    }
}
