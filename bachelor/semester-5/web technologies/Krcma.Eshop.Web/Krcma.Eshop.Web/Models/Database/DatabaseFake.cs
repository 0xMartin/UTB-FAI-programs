using Krcma.Eshop.Web.Models.Entity;
using System.Collections.Generic;

namespace Krcma.Eshop.Web.Models.Database
{
    public static class DatabaseFake
    {
        public static List<CarouselItem> CarouselItems { get; set; }
        public static List<Product> Products { get; set; }

        static DatabaseFake()
        {
            DatabaseInit dbInit = new DatabaseInit();

            CarouselItems = dbInit.GenerateCarouselItems();

            Products = dbInit.GenerateProducts();
        }
    }
}
