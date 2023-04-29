using Krcma.Eshop.Web.Areas.ReturnModule.Entity;
using Krcma.Eshop.Web.Models.Entity;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Krcma.Eshop.Web.Areas.ReturnModule.Models
{
    public class ReturnViewModel
    {
        public int ItemsOnPage { get; set; }

        public int TotalPageCount { get; set; }

        public int CurrentPageNumber { get; set; }

        public IList<Order> UserOrders { get; set; }

        public IList<ReturnProduct> ReturnProducts { get; set; }

    }
}
