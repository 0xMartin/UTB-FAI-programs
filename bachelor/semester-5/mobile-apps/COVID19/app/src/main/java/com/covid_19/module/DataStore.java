package com.covid_19.module;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

public class DataStore {

    private final Context context;

    private static String KEY = "obj";

    public DataStore(Context context) {
        this.context = context;
    }

    public synchronized void store(String name, StorableObject object) {
        SharedPreferences settings = this.context.getSharedPreferences(name, 0);
        SharedPreferences.Editor editor = settings.edit();
        String strData = object.dataToString();
        editor.putString(DataStore.KEY, strData);
        editor.apply();

        Log.d("DataStore", "storing data with the name " + name);
        Log.d("DataStore", strData);
    }

    public synchronized boolean load(String name, StorableObject object) {
        SharedPreferences settings = this.context.getSharedPreferences(name, 0);
        String strData = settings.getString(DataStore.KEY, "");

        Log.d("DataStore", "loading data with the name " + name);

        if(strData.length() == 0) {
            return false;
        }

        object.stringToData(strData);
        Log.d("DataStore", strData);

        return true;
    }

    public interface StorableObject {
        String dataToString();
        void stringToData(String str);
    }


}
