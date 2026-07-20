package org.hiddenmoon.waterreaction;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.widget.LinearLayout;

public final class ResearchRootLayout extends LinearLayout {
    private final Paint gridPaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    public ResearchRootLayout(Context context) {
        super(context);
        init();
    }

    public ResearchRootLayout(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        setWillNotDraw(false);
        setOrientation(VERTICAL);
        setBackgroundColor(UiPalette.PAPER);
        gridPaint.setColor(UiPalette.GRID);
        gridPaint.setStrokeWidth(dp(1));
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        float step = dp(32);
        for (float x = dp(12); x < getWidth(); x += step) {
            canvas.drawLine(x, 0, x, getHeight(), gridPaint);
        }
        for (float y = dp(12); y < getHeight(); y += step) {
            canvas.drawLine(0, y, getWidth(), y, gridPaint);
        }
        Paint accent = new Paint(Paint.ANTI_ALIAS_FLAG);
        accent.setColor(UiPalette.SIGNAL_YELLOW);
        accent.setStrokeWidth(dp(3));
        canvas.drawLine(0, dp(10), dp(72), dp(10), accent);
    }

    private float dp(float value) {
        return value * getResources().getDisplayMetrics().density;
    }
}
