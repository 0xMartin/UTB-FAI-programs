using Krcma.Eshop.Web.Areas.ReturnModule.Entity;
using Krcma.Eshop.Web.Models.Entity;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Krcma.Eshop.Web.Areas.ReturnModule
{
    public static class ReturnModuleShared
    {

        public static string KEY_LIMIT_NON_DIGITAL_PRODUCT = "non_digital_product_limit";

        public static string KEY_LIMIT_DIGITAL_PRODUCT = "digital_product_limit";


        public static List<Property> DefaultPropts()
        {
            List<Property> propts = new List<Property>();

            propts.Add(new Property
            {
                Name = KEY_LIMIT_NON_DIGITAL_PRODUCT,
                Value = "30"
            });

            propts.Add(new Property
            {
                Name = KEY_LIMIT_DIGITAL_PRODUCT,
                Value = "10"
            });

            return propts;
        }

        public static bool VerifyDayLimit(Order order, int dayLimit)
        {
            int days = DayDifference(order);
            return days <= dayLimit;
        }

        public static int DayDifference(Order order)
        {
            DateTime now = DateTime.Now;
            TimeSpan ts = now.Subtract(order.DateTimeCreated.AddDays(order.ShippingDelay));
            return ts.Days;
        }

    }
}
