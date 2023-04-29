package com.covid_19;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.view.MenuItem;
import android.widget.TextView;

import com.covid_19.covid19msg.Covid19CountryData;
import com.covid_19.module.Communication;
import com.github.mikephil.charting.charts.PieChart;
import com.github.mikephil.charting.data.PieData;
import com.github.mikephil.charting.data.PieDataSet;
import com.github.mikephil.charting.data.PieEntry;
import com.github.mikephil.charting.utils.ColorTemplate;
import com.google.android.gms.maps.CameraUpdateFactory;
import com.google.android.gms.maps.GoogleMap;
import com.google.android.gms.maps.OnMapReadyCallback;
import com.google.android.gms.maps.SupportMapFragment;
import com.google.android.gms.maps.model.LatLng;
import com.google.android.gms.maps.model.MarkerOptions;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class StatisticsActivity extends AppCompatActivity implements OnMapReadyCallback {

    public static final String COUNTRY_CODE_KEY = "country_code";

    private String countryCode;

    private GoogleMap mMap = null;

    private TextView textView_update_statistics, textView_country, textView_confirmed, textView_recovered, textView_critical, textView_deaths;
    private PieChart pieChart;

    @RequiresApi(api = Build.VERSION_CODES.O)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_statistics);

        //action bar
        ActionBar actionBar = getSupportActionBar();
        Objects.requireNonNull(actionBar).setDisplayHomeAsUpEnabled(true);

        // Obtain the SupportMapFragment and get notified when the map is ready to be used.
        SupportMapFragment mapFragment = (SupportMapFragment) getSupportFragmentManager()
                .findFragmentById(R.id.map);
        mapFragment.getMapAsync(this);

        //data
        Bundle extras = getIntent().getExtras();
        if (extras != null) {
            this.countryCode = extras.getString(StatisticsActivity.COUNTRY_CODE_KEY);
        } else {
            onBackPressed();
        }

        textView_update_statistics = findViewById(R.id.textView_update_statistics);
        textView_country = findViewById(R.id.textView_country_statistics);
        textView_confirmed = findViewById(R.id.textView_confirmed_statistics);
        textView_recovered = findViewById(R.id.textView_recovered_statistics);
        textView_critical = findViewById(R.id.textView_critical_statistics);
        textView_deaths = findViewById(R.id.textView_deaths_statistics);
        pieChart = findViewById(R.id.chart);
    }

    @Override
    protected void onStart() {
        super.onStart();


        Covid19CountryData covid19CountryData = new Covid19CountryData(this.countryCode) {
            @SuppressLint("SetTextI18n")
            @Override
            public void onResponseEvent(Communication.Message msg) {
                Covid19CountryData data = (Covid19CountryData) msg;
                try {
                    updateStatistics(data);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                //store data
                AppGlobal.getInstance().dataStore.store("statistics_activity", data);
            }

            @Override
            public void error() throws ClassCastException {
                //load data
                Covid19CountryData data = new Covid19CountryData("");
                if(AppGlobal.getInstance().dataStore.load("statistics_activity", data)) {
                    try {
                        updateStatistics(data);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        };

        //update stats
        AppGlobal.getInstance().communication.fetch(covid19CountryData);
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getItemId() == android.R.id.home) {
            onBackPressed();
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    @Override
    public void onMapReady(GoogleMap googleMap) {
        mMap = googleMap;
    }

    private void setMapPosition(Covid19CountryData data, int iterations) {
        if(iterations < 0) return;
        if(mMap == null) {
            final Handler handler = new Handler();
            handler.postDelayed(new Runnable() {
                @Override
                public void run() {
                    setMapPosition(data, iterations - 1);
                }
            }, 200);
        } else {
            LatLng pos = new LatLng(data.latitude, data.longitude);
            mMap.addMarker(new MarkerOptions()
                    .position(pos)
                    .title(data.country));
            mMap.moveCamera(CameraUpdateFactory.newLatLng(pos));
        }
    }

    private void updateStatistics(Covid19CountryData data) throws Exception {
        //values
        String date = data.lastUpdate.replace("T", " ");
        date = date.substring(0, date.indexOf("+"));
        textView_update_statistics.setText(getResources().getString(R.string.last_update) + ": " + date);
        textView_country.setText(data.country + " " + AppGlobal.getInstance().getFlagEmoji(data.countyCode));
        textView_confirmed.setText(AppGlobal.getInstance().getFormatedNumber(data.confirmed));
        textView_recovered.setText(AppGlobal.getInstance().getFormatedNumber(data.recovered));
        textView_critical.setText(AppGlobal.getInstance().getFormatedNumber(data.critical));
        textView_deaths.setText(AppGlobal.getInstance().getFormatedNumber(data.deaths));

        //pie graph
        List<PieEntry> pieEntires = new ArrayList<>();
        pieEntires.add(new PieEntry(data.recovered, "Recovered"));
        pieEntires.add(new PieEntry(data.critical, "Critical"));
        pieEntires.add(new PieEntry(data.confirmed, "Confirmed"));
        pieEntires.add(new PieEntry(data.deaths, "Deaths"));
        PieDataSet dataSet = new PieDataSet(pieEntires,"");
        dataSet.setColors(ColorTemplate.MATERIAL_COLORS);
        PieData gdata = new PieData(dataSet);
        gdata.setValueTextSize(14f);
        //get the pieChart
        pieChart.setData(gdata);
        pieChart.setDrawHoleEnabled(false);
        pieChart.setEntryLabelTextSize(20f);
        pieChart.getDescription().setEnabled(false);
        pieChart.getLegend().setEnabled(false);
        pieChart.animateXY(1000, 1000);
        pieChart.setContentDescription("");
        pieChart.invalidate();

        //map
        setMapPosition(data, 10);
    }

}