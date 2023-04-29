package com.covid_19.module;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.LinearGradient;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Shader;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.View;

import com.covid_19.R;

import java.util.ArrayList;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;

@SuppressLint("ViewConstructor")
public class BackgroundAnimation extends View {

    final int particleCount;
    int fps;

    final Resources res;
    Timer timer = null;

    Bitmap particleBitmap;
    Paint gradientPaint;
    DisplayMetrics displayMetrics;

    final List<Particle> particleList;

    public BackgroundAnimation(Context context, int particleCount, int fps) {
        super(context);
        this.res = context.getResources();
        this.particleList = new ArrayList<>();

        this.particleCount = particleCount;
        this.fps = fps;
    }

    public void init() {
        this.gradientPaint = new Paint();

        displayMetrics = new DisplayMetrics();
        ((Activity) this.getContext()).getWindowManager()
                .getDefaultDisplay()
                .getMetrics(displayMetrics);

        //gradient
        Shader shader = new LinearGradient(
                0,
                (int)(displayMetrics.heightPixels * 0.1),
                0,
                (int)(displayMetrics.heightPixels * 1.4),
                res.getColor(R.color.black),
                res.getColor(R.color.dark_red),
                Shader.TileMode.CLAMP
        );
        this.gradientPaint.setShader(shader);

        //particle image
        this.particleBitmap = BitmapFactory.decodeResource(this.res, R.drawable.particle);

        //create particles
        final float fps_ratio = (float)(20.0 / fps);

        this.particleList.clear();
        for(int i = 0; i < this.particleCount; ++i) {
            this.particleList.add(new Particle(
                    this.particleBitmap,
                    (int) ((Math.random() * 0.8 + 0.1) * displayMetrics.widthPixels),
                    (int) ((Math.random() * (1.0 / this.particleCount) + (1.0 / this.particleCount) * i) * displayMetrics.heightPixels),
                    (float) (Math.random() * 5.0 - 2.5) * fps_ratio,
                    (float) (Math.random() * 2.8 + 2.5) * fps_ratio,
                    (int) (Math.random() * 250 + 150)
            ));
        }
    }

    public synchronized void spawnNewParticle() {
        final float fps_ratio = (float)(20.0 / fps);

        this.particleList.add(new Particle(
                this.particleBitmap,
                (int) ((Math.random() * 0.8 + 0.1) * displayMetrics.widthPixels),
                -300,
                (float) (Math.random() * 5.0 - 2.5) * fps_ratio,
                (float) (Math.random() * 2.8 + 2.5) * fps_ratio,
                (int) (Math.random() * 250 + 150)
        ));
    }

    public synchronized void run() {
        if(this.timer == null) {
            this.timer = new Timer();
            this.timer.schedule(new TimerTask() {
                @Override
                public void run() {
                    update();
                }
            }, 0, 1000 / this.fps);
            Log.d("BgAnimation", "Run");
        }
    }

    public synchronized void stop() {
        if(this.timer != null) {
            this.timer.cancel();
            this.timer = null;
            Log.d("BgAnimation", "Stop");
        }
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        //draw gradient
        canvas.drawRect(0, 0, this.getWidth(), this.getHeight(), this.gradientPaint);

        //draw all particles
        for(Particle p : this.particleList) {
            if(p != null) {
                p.draw(canvas);
            }
        }
    }

    private void update() {
        //update all
        boolean status;
        for(int i = 0; i < this.particleList.size(); ++i) {
            status = this.particleList.get(i).computePosition(
                    this.displayMetrics.widthPixels, this.displayMetrics.heightPixels);
            if(!status) {
                //remove this particle
                this.particleList.remove(i);
                i -= 1;
                //add new particle
                spawnNewParticle();
            }
        }

        //request repaint
        super.invalidate();
    }

    private static class Particle {

        private Bitmap bitmap;
        private float x, y;   //current position
        private float vx, vy; //velocity
        private final int radius;

        public Particle(Bitmap _bitmap, int _x, int _y, float _vx, float _vy, int _size) {
            this.bitmap = getResizedBitmap(_bitmap, _size, _size);
            this.x = _x;
            this.y = _y;
            this.vx = _vx;
            this.vy = _vy;
            this.radius = _size / 2;
        }

        public void draw(Canvas canvas) {
            canvas.drawBitmap(this.bitmap, (int)this.x - radius, (int)this.y - radius, null);
        }

        public boolean computePosition(int _screenWidth, int _screenHeight) {
            this.x += this.vx;
            this.y += this.vy;

            if((this.x + this.radius < 0) || (this.x - this.radius > _screenWidth)) {
                return false;
            }
            if(this.y - this.radius > _screenHeight) {
                return false;
            }

            return true;
        }

        public Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
            int width = bm.getWidth();
            int height = bm.getHeight();
            float scaleWidth = ((float) newWidth) / width;
            float scaleHeight = ((float) newHeight) / height;
            Matrix matrix = new Matrix();
            matrix.postScale(scaleWidth, scaleHeight);
            return Bitmap.createBitmap(
                    bm, 0, 0, width, height, matrix, false);
        }

    }

}
