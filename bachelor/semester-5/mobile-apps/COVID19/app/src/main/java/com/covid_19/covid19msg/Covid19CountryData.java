package com.covid_19.covid19msg;

import android.util.Log;

import com.covid_19.module.Communication;
import com.covid_19.module.DataStore;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.Serializable;

public class Covid19CountryData extends Communication.Message  {

    /***DATA*****************************************************************/
    public String country;
    public int confirmed;
    public int recovered;
    public int critical;
    public int deaths;
    public double latitude;
    public double longitude;
    public String lastChange;
    public String lastUpdate;
    /***DATA*****************************************************************/


    public String countyCode;

    public Covid19CountryData(String countyCode) {
        this.countyCode = countyCode;
    }

    @Override
    public String getAPIURL() {
        return "https://covid-19-data.p.rapidapi.com/country/code?code=" + this.countyCode;
    }

    @Override
    public void parseJSON(String json) {
        try {

            JSONArray array = new JSONArray(json);
            JSONObject jObject = array.getJSONObject(0);
            countyCode = jObject.getString("code");
            country = jObject.getString("country");
            confirmed = jObject.getInt("confirmed");
            recovered = jObject.getInt("recovered");
            critical = jObject.getInt("critical");
            deaths = jObject.getInt("deaths");
            latitude = jObject.getDouble("latitude");
            longitude = jObject.getDouble("longitude");
            lastChange = jObject.getString("lastChange");
            lastUpdate = jObject.getString("lastUpdate");

        } catch (JSONException e) {
            Log.d("Covid19CountryData Exception", e.getMessage());
        }
    }

    @Override
    public void error() {

    }



    @Override
    public String dataToString() {
        JSONObject jObject = new JSONObject();
        try {
            jObject.put("code", countyCode);
            jObject.put("country", country);
            jObject.put("confirmed", confirmed);
            jObject.put("recovered", recovered);
            jObject.put("critical", critical);
            jObject.put("deaths", deaths);
            jObject.put("latitude", latitude);
            jObject.put("longitude", longitude);
            jObject.put("lastChange", lastChange);
            jObject.put("lastUpdate", lastUpdate);
        } catch (JSONException e) {
            e.printStackTrace();
        }

        JSONArray array = new JSONArray();
        array.put(jObject);
        return array.toString();
    }

    @Override
    public void stringToData(String str) {
        this.parseJSON(str);
    }
}
