package com.covid_19;

import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.TableRow;
import android.widget.TextView;

import com.covid_19.covid19msg.Covid19CountryData;
import com.covid_19.module.Communication;

import java.util.Objects;

public class HomeActivity extends AppCompatActivity {

    private boolean isVisible = false;

    private TextView textView_country, textView_confirmed, textView_update, textView_recovered, textView_critical, textView_deaths;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home);
        Objects.requireNonNull(getSupportActionBar()).hide();

        //bg animation
        final FrameLayout frameLayout = findViewById(R.id.background_home);
        if(AppGlobal.getInstance().animation != null) {
            frameLayout.addView(AppGlobal.getInstance().animation, 0);
        }


        Button btn;

        //button_statistics
        btn = (Button) findViewById(R.id.button_statistics);
        btn.setOnClickListener(view -> {
            Intent intent = new Intent(HomeActivity.this, StatisticsActivity.class);
            intent.putExtra(StatisticsActivity.COUNTRY_CODE_KEY, AppGlobal.Setting.homeCountryCode);
            startActivity(intent);
        });

        //button_global
        btn = (Button) findViewById(R.id.button_global);
        btn.setOnClickListener(view -> {
            Intent intent = new Intent(getApplicationContext(), GlobalActivity.class);
            startActivity(intent);
        });

        //button_settings
        btn = (Button) findViewById(R.id.button_settings);
        btn.setOnClickListener(view -> {
            Intent intent = new Intent(getApplicationContext(), SettingsActivity.class);
            startActivity(intent);
        });

        textView_country = findViewById(R.id.textView_country);
        textView_confirmed = findViewById(R.id.textView_confirmed);
        textView_update = findViewById(R.id.textView_update);
        textView_recovered = findViewById(R.id.textView_recovered);
        textView_critical = findViewById(R.id.textView_critical);
        textView_deaths = findViewById(R.id.textView_deaths);
    }

    @Override
    protected void onStart() {
        super.onStart();
        if(AppGlobal.getInstance().animation != null) {
            AppGlobal.getInstance().animation.run();
        }

        this.isVisible = true;

        //update stats
        update();

        //(Settings) apply visibility
        final TextView textView_update = findViewById(R.id.textView_update);
        textView_update.setVisibility(AppGlobal.Setting.updateTime_visible ? View.VISIBLE : View.GONE);

        final TableRow tableRow_confirmed = findViewById(R.id.tableRow_confirmed);
        tableRow_confirmed.setVisibility(AppGlobal.Setting.confirmed_visible ? View.VISIBLE : View.GONE);

        final TableRow tableRow_recovered = findViewById(R.id.tableRow_recovered);
        tableRow_recovered.setVisibility(AppGlobal.Setting.recovered_visible ? View.VISIBLE : View.GONE);

        final TableRow tableRow_critical = findViewById(R.id.tableRow_critical);
        tableRow_critical.setVisibility(AppGlobal.Setting.critical_visible ? View.VISIBLE : View.GONE);

        final TableRow tableRow_deaths = findViewById(R.id.tableRow_deaths);
        tableRow_deaths.setVisibility(AppGlobal.Setting.deaths_visible ? View.VISIBLE : View.GONE);
    }

    @Override
    protected void onPause() {
        super.onPause();
        if(AppGlobal.getInstance().animation != null) {
            AppGlobal.getInstance().animation.stop();
        }

        this.isVisible = false;
    }

    @SuppressLint("SetTextI18n")
    private void showData(Covid19CountryData data) throws Exception {
        textView_country.setText(data.country + " " + AppGlobal.getInstance().getFlagEmoji(data.countyCode));
        String date = data.lastUpdate.replace("T", " ");
        date = date.substring(0, date.indexOf("+"));
        textView_update.setText(getResources().getString(R.string.last_update) + ": " + date);
        textView_confirmed.setText(AppGlobal.getInstance().getFormatedNumber(data.confirmed));
        textView_recovered.setText(AppGlobal.getInstance().getFormatedNumber(data.recovered));
        textView_critical.setText(AppGlobal.getInstance().getFormatedNumber(data.critical));
        textView_deaths.setText(AppGlobal.getInstance().getFormatedNumber(data.deaths));
    }

    public void update() {
        Covid19CountryData covid19CountryData = new Covid19CountryData(AppGlobal.Setting.homeCountryCode) {
            @SuppressLint("SetTextI18n")
            @Override
            public void onResponseEvent(Communication.Message msg) {
                Covid19CountryData data = (Covid19CountryData) msg;
                try {
                    showData(data);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                //store data
                AppGlobal.getInstance().dataStore.store("home_activity", data);
            }

            @Override
            public void error() throws ClassCastException {
                //load data
                Covid19CountryData data = new Covid19CountryData("");
                if(AppGlobal.getInstance().dataStore.load("home_activity", data)) {
                    try {
                        showData(data);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        };
        AppGlobal.getInstance().communication.fetch(covid19CountryData);

        //next update
        if(this.isVisible) {
            final Handler handler = new Handler();
            handler.postDelayed(new Runnable() {
                @Override
                public void run() {
                    if(isVisible) {
                        Log.d("HomeActivity", "Refresh");
                        update();
                    }
                }
            }, 1000 * 120);
        }
    }

}