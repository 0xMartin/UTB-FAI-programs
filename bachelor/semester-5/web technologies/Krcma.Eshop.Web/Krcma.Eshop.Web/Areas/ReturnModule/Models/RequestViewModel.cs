using Krcma.Eshop.Web.Models.Entity;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Krcma.Eshop.Web.Areas.ReturnModule.Models
{
    public class RequestViewModel
    {

        public OrderItem OrderItem { get; set; }

        public bool Approved { get; set; }

    }
}
