using Krcma.Eshop.Web.Models.Entity;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Krcma.Eshop.Web.Models.ViewModels
{
    public class ProductsViewModel
    {

        public int ItemsOnPage { get; set; }

        public int TotalPageCount { get; set; }

        public int CurrentPageNumber { get; set; }

        public IList<Product> Products { get; set; }

    }
}
