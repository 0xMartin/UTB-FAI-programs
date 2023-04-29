using Krcma.Eshop.Web.Models.Entity;
using System.Collections.Generic;

namespace Krcma.Eshop.Web.Models.ViewModels
{
    public class IndexViewModel
    {
        public IList<CarouselItem> CarouselItems { get; set; }
        public IList<Product> Products { get; set; }
    }
}
