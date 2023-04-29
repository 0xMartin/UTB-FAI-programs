using Krcma.Eshop.Web.Areas.ReturnModule.Entity;
using Krcma.Eshop.Web.Areas.ReturnModule.Models;
using Krcma.Eshop.Web.Models.ApplicationServices.Abstraction;
using Krcma.Eshop.Web.Models.Database;
using Krcma.Eshop.Web.Models.Entity;
using Krcma.Eshop.Web.Models.Identity;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Krcma.Eshop.Web.Areas.ReturnModule.Controllers
{

    [Area("ReturnModule")]
    [Authorize(Roles = nameof(Roles.Customer))]
    public class ReturnController : Controller
    {

        readonly EshopDbContext eshopDbContext;
        ISecurityApplicationService iSecure;

        public ReturnController(ISecurityApplicationService iSecure, EshopDbContext eshopDb)
        {
            this.iSecure = iSecure;
            this.eshopDbContext = eshopDb;
        }

        // GET: ReturnController
        public async Task<IActionResult> Index(int PAGE)
        {
            if (User.Identity.IsAuthenticated)
            {
                User currentUser = await iSecure.GetCurrentUser(User);
                if (currentUser != null)
                {
                    ReturnViewModel rvm = new ReturnViewModel();

                    rvm.ItemsOnPage = 5;
                    rvm.CurrentPageNumber = Math.Max(PAGE, 1);

                    rvm.UserOrders = await this.eshopDbContext.Orders.Where(or => or.UserId == currentUser.Id)
                                                                          .Include(o => o.User)
                                                                          .Include(o => o.OrderItems)
                                                                          .ThenInclude(oi => oi.Product)
                                                                          .ToListAsync();


                    rvm.ReturnProducts = this.eshopDbContext.ReturnProducts.ToList();

                    int orderItemsCount = 0;
                    foreach (var order in rvm.UserOrders) orderItemsCount += order.OrderItems.Count;

                    rvm.TotalPageCount = (int)Math.Ceiling((double)orderItemsCount / rvm.ItemsOnPage);

                    return View(rvm);
                }
            }
            return NotFound();
        }

        public async Task<IActionResult> RequestProductReturn(int orderItemID)
        {
            if (User.Identity.IsAuthenticated)
            {
                User currentUser = await iSecure.GetCurrentUser(User);

                OrderItem orderItem = this.eshopDbContext.OrderItems.Where(o => o.ID == orderItemID).Include(o => o.Product).First();

                if (orderItem != null)
                {
                    if(orderItem.Product != null) { 

                    ReturnProduct rp = new ReturnProduct();
                    //rp.ID = this.eshopDbContext.ReturnProducts.Count() + 1;
                    rp.OrderItemID = orderItem.ID;
                    rp.Approved = false;
                    rp.Processed = false;

                        if (!eshopDbContext.ReturnProducts.Any(r => r.OrderItemID == rp.OrderItemID))
                        {
                            //automaticke zpracovani (jen pro digitalni zbozi)
                            if (orderItem.Product.DigitalProduct)
                            {
                                Order order = this.eshopDbContext.Orders.FirstOrDefault(o => o.ID == orderItem.OrderID);
                                if (order != null)
                                {
                                    Property propt = this.eshopDbContext.ReturnProperties.
                                        FirstOrDefault(p => p.Name.Equals(ReturnModuleShared.KEY_LIMIT_DIGITAL_PRODUCT));
                                    if (propt != null)
                                    {
                                        if (Int32.TryParse(propt.Value, out int dayLimit))
                                        {
                                            rp.Approved = ReturnModuleShared.VerifyDayLimit(order, dayLimit);
                                            rp.Processed = true;
                                        }
                                    }
                                }
                            }
                            this.eshopDbContext.ReturnProducts.Add(rp);
                            eshopDbContext.SaveChanges();
                        }

                        RequestViewModel rvm = new RequestViewModel()
                        {
                            OrderItem = orderItem,
                            Approved = rp.Approved
                        };

                        return View(rvm);
                    }
                }
            }

            return NotFound();
        }

    }

}
