package com.covid_19.covid19msg;

import android.util.Log;

import com.covid_19.module.Communication;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

public class Covid19TotalData extends Communication.Message  {

    /***DATA*****************************************************************/
    public int confirmed;
    public int recovered;
    public int critical;
    public int deaths;
    public String lastChange;
    public String lastUpdate;
    /***DATA*****************************************************************/

    @Override
    public String getAPIURL() {
        return "https://covid-19-data.p.rapidapi.com/totals";
    }

    @Override
    public void parseJSON(String json) {
        try {

            JSONArray array = new JSONArray(json);
            JSONObject jObject = array.getJSONObject(0);
            confirmed = jObject.getInt("confirmed");
            recovered = jObject.getInt("recovered");
            critical = jObject.getInt("critical");
            deaths = jObject.getInt("deaths");
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
            jObject.put("confirmed", confirmed);
            jObject.put("recovered", recovered);
            jObject.put("critical", critical);
            jObject.put("deaths", deaths);
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
