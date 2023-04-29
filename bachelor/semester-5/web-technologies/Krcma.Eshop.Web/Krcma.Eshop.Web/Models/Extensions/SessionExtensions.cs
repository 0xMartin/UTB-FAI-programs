using System;
using Microsoft.AspNetCore.Http;
using Newtonsoft.Json;

/// <summary>
///extensions for session
/// </summary>
public static class SessionExtensions
{

    public static double? GetDouble(this ISession session, string key)
    {
        var data = session.Get(key);
        if (data == null)
        {
            return null;
        }
        return BitConverter.ToDouble(data, 0);
    }

    public static void SetDouble(this ISession session, string key, double value)
    {
        session.Set(key, BitConverter.GetBytes(value));
    }


    //for objects such as List, arrays etc.
    public static T GetObject<T>(this ISession session, string key)
    {
        var data = session.GetString(key);
        if (data == null)
        {
            return default(T);
        }
        return JsonConvert.DeserializeObject<T>(data);
    }

    public static void SetObject(this ISession session, string key, object value)
    {
        session.SetString(key, JsonConvert.SerializeObject(value));
    }
}
