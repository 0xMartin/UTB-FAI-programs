using Krcma.Eshop.Web.Areas.ReturnModule.Entity;
using Krcma.Eshop.Web.Models.Database;
using Krcma.Eshop.Web.Models.Entity;
using Krcma.Eshop.Web.Models.Identity;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Krcma.Eshop.Web.Areas.ReturnModule.Controllers
{

    [Area("ReturnModule")]
    [Authorize(Roles = nameof(Roles.Admin) + ", " + nameof(Roles.Manager))]
    public class Manage : Controller
    {

        readonly EshopDbContext eshopDbContext;
        private readonly ILogger logger;

        public Manage(EshopDbContext eshopDb, ILogger<Manage> logger)
        {
            this.eshopDbContext = eshopDb;
            this.logger = logger;
        }

        public async Task<IActionResult> Index()
        {
            IList<ReturnProduct> returnProducts = await this.eshopDbContext.ReturnProducts.Include(rp => rp.OrderItem)
                                                                                          .ThenInclude(o => o.Order).ToListAsync();

            foreach (var returnProduct in returnProducts)
            {
                if (returnProduct.OrderItem == null) continue;
                if (returnProduct.OrderItem.Order == null) continue;

                Property propt = this.eshopDbContext.ReturnProperties.
                                       FirstOrDefault(p => p.Name.Equals(ReturnModuleShared.KEY_LIMIT_NON_DIGITAL_PRODUCT));
                if (propt != null)
                {
                    if (Int32.TryParse(propt.Value, out int dayLimit))
                    {
                        returnProduct.RemainingDays = dayLimit - ReturnModuleShared.DayDifference(returnProduct.OrderItem.Order);
                    }
                }
            }

            return View(returnProducts);
        }

        public IActionResult EditLimits()
        {
            IList<Property> propts = this.eshopDbContext.ReturnProperties.ToList();

            return View(propts);
        }

        [HttpPost]
        public async Task<IActionResult> EditLimits(IList<Property> propts)
        {
            if (propts != null)
            {
                foreach (Property propt in propts)
                {
                    Property proptFromDB = this.eshopDbContext.ReturnProperties.FirstOrDefault(p => p.Name.Equals(propt.Name));
                    proptFromDB.Value = propt.Value;
                    logger.LogInformation(proptFromDB.Name + " set on " + proptFromDB.Value);
                }

                await this.eshopDbContext.SaveChangesAsync();
            }

            return View(propts);
        }

        public async Task<IActionResult> Approve(int ID)
        {
            ReturnProduct rp = this.eshopDbContext.ReturnProducts.FirstOrDefault(rp => rp.ID == ID);
            if (rp != null)
            {
                rp.Approved = true;
                rp.Processed = true;
                await eshopDbContext.SaveChangesAsync();
            }
            return RedirectToAction(nameof(Manage.Index));
        }

        public async Task<IActionResult> Disapprove(int ID)
        {
            ReturnProduct rp = this.eshopDbContext.ReturnProducts.FirstOrDefault(rp => rp.ID == ID);
            if (rp != null)
            {
                rp.Approved = false;
                rp.Processed = true;
                await this.eshopDbContext.SaveChangesAsync();
            }
            return RedirectToAction(nameof(Manage.Index));
        }

        public IActionResult ShippingDelay()
        {
            IList<Order> orders = this.eshopDbContext.Orders.ToList();

            return View(orders);
        }

        public IActionResult SetShippingDelay(int ID)
        {
            Order order = this.eshopDbContext.Orders.FirstOrDefault(o => o.ID == ID);
            return View(order);
        }

        [HttpPost]
        public async Task<IActionResult> SetShippingDelay(Order ord)
        {
            Order order = this.eshopDbContext.Orders.FirstOrDefault(o => o.ID == ord.ID);
            if (order != null)
            {
                if (ord.ShippingDelay >= 0)
                {
                    order.ShippingDelay = ord.ShippingDelay;
                    await this.eshopDbContext.SaveChangesAsync();
                    logger.LogInformation(order.OrderNumber + " sent delay " + order.ShippingDelay);
                }
            }

            return RedirectToAction(nameof(Manage.ShippingDelay));
        }

    }
}
