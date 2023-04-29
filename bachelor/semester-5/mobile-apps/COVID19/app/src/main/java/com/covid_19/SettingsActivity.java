package com.covid_19;

import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Spinner;
import android.widget.Switch;
import android.widget.TextView;

import com.covid_19.covid19msg.Covid19CountryList;
import com.covid_19.module.Communication;

import java.util.Objects;

public class SettingsActivity extends AppCompatActivity {

    private Spinner spiner_countryCode;

    @SuppressLint("UseSwitchCompatOrMaterialCode")
    private Switch switch_update, switch_confirmed, switch_recovered, switch_critical, switch_deaths;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_settings);

        //action bar
        ActionBar actionBar = getSupportActionBar();
        Objects.requireNonNull(actionBar).setDisplayHomeAsUpEnabled(true);

        //update switch status
        switch_update = findViewById(R.id.switch_update);
        switch_update.setChecked(AppGlobal.Setting.updateTime_visible);
        switch_update.setOnCheckedChangeListener((compoundButton, b) -> saveChanges());

        switch_confirmed = findViewById(R.id.switch_confirmed);
        switch_confirmed.setChecked(AppGlobal.Setting.confirmed_visible);
        switch_confirmed.setOnCheckedChangeListener((compoundButton, b) -> saveChanges());

        switch_recovered = findViewById(R.id.switch_recovered);
        switch_recovered.setChecked(AppGlobal.Setting.recovered_visible);
        switch_recovered.setOnCheckedChangeListener((compoundButton, b) -> saveChanges());

        switch_critical = findViewById(R.id.switch_critical);
        switch_critical.setChecked(AppGlobal.Setting.critical_visible);
        switch_critical.setOnCheckedChangeListener((compoundButton, b) -> saveChanges());

        switch_deaths = findViewById(R.id.switch_deaths);
        switch_deaths.setChecked(AppGlobal.Setting.deaths_visible);
        switch_deaths.setOnCheckedChangeListener((compoundButton, b) -> saveChanges());
    }

    @Override
    public void onStart() {
        super.onStart();

        //update country list (spinner)
        AppGlobal.getInstance().communication.fetch(covid19CountryAll);
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getItemId() == android.R.id.home) {
            onBackPressed();
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    public void saveChanges() {
        //update changes
        AppGlobal.Setting.homeCountryCode = Objects.requireNonNull(((Covid19CountryList.Country)spiner_countryCode.getSelectedItem())).code;
        AppGlobal.Setting.updateTime_visible = switch_update.isChecked();
        AppGlobal.Setting.confirmed_visible = switch_confirmed.isChecked();
        AppGlobal.Setting.recovered_visible = switch_recovered.isChecked();
        AppGlobal.Setting.critical_visible = switch_critical.isChecked();
        AppGlobal.Setting.deaths_visible = switch_deaths.isChecked();

        //save to storage
        AppGlobal.getInstance().storeSettings();
    }


    final Covid19CountryList covid19CountryAll = new Covid19CountryList() {
        @Override
        public void onResponseEvent(Communication.Message msg) {
            Covid19CountryList data = (Covid19CountryList) msg;
            try {
                initSpinner(data);
            } catch (Exception e) {
                e.printStackTrace();
            }
            //store data
            AppGlobal.getInstance().dataStore.store("settings_activity", data);
        }
        @Override
        public void error() throws ClassCastException {
            //load data
            Covid19CountryList data = new Covid19CountryList();
            if(AppGlobal.getInstance().dataStore.load("settings_activity", data)) {
                try {
                    initSpinner(data);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    };

    private void initSpinner(Covid19CountryList data) throws Exception {
        //create adapter for spinner
        ArrayAdapter<Covid19CountryList.Country> adapter = new ArrayAdapter<Covid19CountryList.Country>(
                this, R.layout.item_country, R.id.name_country_item, data.countries) {
            @Override
            public View getView(int position, View convertView, ViewGroup parent) {
                return customView(position, convertView, parent);
            }

            @Override
            public View getDropDownView(int position, View convertView, ViewGroup parent) {
                return customView(position, convertView, parent);
            }

            private View customView(int position, View convertView, ViewGroup parent) {
                if (convertView == null) {
                    convertView = LayoutInflater.from(getContext()).inflate(R.layout.item_country, parent, false);
                }
                Covid19CountryList.Country country = getItem(position);
                final TextView flag = convertView.findViewById(R.id.flag_country_item);
                final TextView name = convertView.findViewById(R.id.name_country_item);
                flag.setText(AppGlobal.getInstance().getFlagEmoji(country.code));
                name.setText(country.name);
                return convertView;
            }
        };

        //set adapter
        spiner_countryCode = findViewById(R.id.spiner_countryCode);
        spiner_countryCode.setAdapter(adapter);

        //set change listener
        spiner_countryCode.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                saveChanges();
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        //selected item
        int i = -1;
        String cc = AppGlobal.Setting.homeCountryCode.toLowerCase();
        for(Covid19CountryList.Country c : data.countries) {
            if(c.code.toLowerCase().equals(cc)) {
                spiner_countryCode.setSelection(i + 1);
                break;
            }
            ++i;
        }
        if(i == -1 && data.countries.size() > 0) {
            spiner_countryCode.setSelection(0);
        }
    }

}