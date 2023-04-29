package com.covid_19;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

import com.covid_19.module.BackgroundAnimation;
import com.covid_19.module.Communication;
import com.covid_19.module.DataStore;

public class AppGlobal {

    private final Context context;

    public BackgroundAnimation animation;

    public Communication communication;

    public DataStore dataStore;

    @SuppressLint("StaticFieldLeak")
    private static AppGlobal appGlobal = null;


    private AppGlobal(Context context) {
        this.context = context;

        //Communication
        this.communication = new Communication(context);

        //DataBackup
        this.dataStore = new DataStore(context);

        //BackgroundAnimation
        this.animation = new BackgroundAnimation(context, 5, 20);
        this.animation.init();

        //load settings from storage
        this.loadSettings();
    }

    public static void init(Context context) {
        if(AppGlobal.appGlobal == null) {
            AppGlobal.appGlobal = new AppGlobal(context);
            Log.d("APP", "init done");
        }
    }

    public static AppGlobal getInstance() {
        return AppGlobal.appGlobal;
    }

    public String getFormatedNumber(int number) {
        String num = Integer.toString(number);
        StringBuilder builder = new StringBuilder();
        for(int i = 0; i < num.length(); ++i) {
            if(i % 3 == 0) {
                builder.append(" ");
            }
            builder.append(num.charAt(num.length() - 1 - i));
        }
        return builder.reverse().toString();
    }

    public String getFlagEmoji(String countryCode) {
        if(countryCode.length() == 0) return "";

        countryCode = countryCode.toUpperCase();
        int firstLetter = Character.codePointAt(countryCode, 0) - 0x41 + 0x1F1E6;
        int secondLetter = Character.codePointAt(countryCode, 1) - 0x41 + 0x1F1E6;
        return new String(Character.toChars(firstLetter)) + new String(Character.toChars(secondLetter));
    }

    public synchronized void storeSettings() {
        SharedPreferences settings = this.context.getSharedPreferences("SETTINGS", 0);
        SharedPreferences.Editor editor = settings.edit();

        editor.putBoolean("updateTime_visible", Setting.updateTime_visible);
        editor.putBoolean("confirmed_visible", Setting.confirmed_visible);
        editor.putBoolean("recovered_visible", Setting.recovered_visible);
        editor.putBoolean("critical_visible", Setting.critical_visible);
        editor.putBoolean("deaths_visible", Setting.deaths_visible);
        editor.putString("homeCountryCode", Setting.homeCountryCode);

        editor.apply();
    }

    public synchronized void loadSettings() {
        // Get from the SharedPreferences
        SharedPreferences settings = this.context.getSharedPreferences("SETTINGS", 0);
        Setting.updateTime_visible = settings.getBoolean("updateTime_visible", true);
        Setting.confirmed_visible = settings.getBoolean("confirmed_visible", true);
        Setting.recovered_visible = settings.getBoolean("recovered_visible", true);
        Setting.critical_visible = settings.getBoolean("critical_visible", true);
        Setting.deaths_visible = settings.getBoolean("deaths_visible", true);
        Setting.homeCountryCode = settings.getString("homeCountryCode", "cz");
    }

    public abstract static class Setting {
        public static String homeCountryCode = "cz";
        public static boolean updateTime_visible = true;
        public static boolean confirmed_visible = true;
        public static boolean recovered_visible = true;
        public static boolean critical_visible = true;
        public static boolean deaths_visible = true;
    }

}
