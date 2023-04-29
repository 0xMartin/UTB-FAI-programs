package com.covid_19.module;

import android.content.Context;
import android.util.Log;

import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;

import java.util.HashMap;
import java.util.Map;

public class Communication {

    private final RequestQueue queue;

    public Communication(Context context) {
        this.queue = Volley.newRequestQueue(context);
    }

    public void fetch(Communication.Message msg) {
        if(msg == null) return;

        String url = msg.getAPIURL();

        //request a string response from the provided URL.
        StringRequest stringRequest = new StringRequest(Request.Method.GET, url,
                response -> {
                    Log.d("Communication", response);
                    msg.parseJSONMain(response);
                },
                error -> {
                    Log.d("Communication ", "API request error");
                    try {
                        msg.error();
                    } catch (ClassCastException ignored) {
                    }
                }) {
            @Override
            public Map<String, String> getHeaders() throws AuthFailureError {
                Map<String, String> params = new HashMap<String, String>();
                params.put("x-rapidapi-host", "covid-19-data.p.rapidapi.com");
                params.put("x-rapidapi-key", "1272c04a02mshb63e7ff4ff037f0p12e83fjsn4a9bacb51b8e");
                return params;
            }
        };

        //add the request to the RequestQueue.
        queue.add(stringRequest);
    }

    public static abstract class Message implements DataStore.StorableObject {

        public abstract String getAPIURL();
        public abstract void parseJSON(String json);
        public abstract void error() throws ClassCastException;

        private void parseJSONMain(String json) {
            this.parseJSON(json);
            this.onResponseEvent(this);
        }

        public void onResponseEvent(Message msg) {
        }
    }

}
