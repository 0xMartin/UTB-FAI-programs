using Krcma.Eshop.Web.Controllers;
using Krcma.Eshop.Web.Models.ApplicationServices.Abstraction;
using Krcma.Eshop.Web.Models.Database;
using Krcma.Eshop.Web.Models.Entity;
using Krcma.Eshop.Web.Models.Identity;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Krcma.Eshop.Web.Areas.Customer.Controllers
{
    [Area("Customer")]
    [Authorize(Roles = nameof(Roles.Customer))]
    public class CustomerOrderNotCartController : Controller
    {
        const string totalPriceString = "TotalPrice";
        const string orderItemsString = "OrderItems";


        ISecurityApplicationService iSecure;
        EshopDbContext EshopDbContext;
        public CustomerOrderNotCartController(ISecurityApplicationService iSecure, EshopDbContext eshopDBContext)
        {
            this.iSecure = iSecure;
            EshopDbContext = eshopDBContext;
        }


        [HttpPost]
        public double AddOrderItemsToSession(int? productId, int? quantity)
        {
            double totalPrice = 0;
            if (HttpContext.Session.IsAvailable)
            {
                totalPrice = HttpContext.Session.GetDouble(totalPriceString).GetValueOrDefault();
            }


            Product product = EshopDbContext.Products.Where(prod => prod.ID == productId).FirstOrDefault();

            if (product != null)
            {
                OrderItem orderItem = new OrderItem()
                {
                    ProductID = product.ID,
                    Product = product,
                    Amount = 1,
                    Price = product.Price   //zde pozor na datový typ -> pokud máte Price v obou případech double nebo decimal, tak je to OK. Mě se bohužel povedlo mít to jednou jako decimal a jednou jako double. Nejlepší je datový typ změnit v databázi/třídě, tak to prosím udělejte.
                };

                if (HttpContext.Session.IsAvailable)
                {

                    List<OrderItem> orderItems = HttpContext.Session.GetObject<List<OrderItem>>(orderItemsString);
                    OrderItem orderItemInSession = null;
                    if (orderItems != null)
                        orderItemInSession = orderItems.Find(oi => oi.ProductID == orderItem.ProductID);
                    else
                        orderItems = new List<OrderItem>();


                    if (orderItemInSession != null)
                    {
                        ++orderItemInSession.Amount;
                        orderItemInSession.Price += orderItem.Product.Price;   //zde pozor na datový typ -> pokud máte Price v obou případech double nebo decimal, tak je to OK. Mě se bohužel povedlo mít to jednou jako decimal a jednou jako double. Nejlepší je datový typ změnit v databázi/třídě, tak to prosím udělejte.
                    }
                    else
                    {
                        orderItems.Add(orderItem);
                    }


                    HttpContext.Session.SetObject(orderItemsString, orderItems);

                    totalPrice += orderItem.Product.Price;
                    HttpContext.Session.SetDouble(totalPriceString, totalPrice);
                }
            }


            return totalPrice;
        }


        public async Task<IActionResult> ApproveOrderInSession()
        {
            if (HttpContext.Session.IsAvailable)
            {


                double totalPrice = 0;
                List<OrderItem> orderItems = HttpContext.Session.GetObject<List<OrderItem>>(orderItemsString);
                if (orderItems != null)
                {
                    foreach (OrderItem orderItem in orderItems)
                    {
                        totalPrice += orderItem.Product.Price * orderItem.Amount;
                        orderItem.Product = null; //zde musime nullovat referenci na produkt, jinak by doslo o pokus jej znovu vlozit do databaze
                    }


                    User currentUser = await iSecure.GetCurrentUser(User);

                    Order order = new Order()
                    {
                        OrderNumber = Convert.ToBase64String(Guid.NewGuid().ToByteArray()),
                        TotalPrice = totalPrice,
                        OrderItems = orderItems,
                        UserId = currentUser.Id
                    };



                    //We can add just the order; order items will be added automatically.
                    await EshopDbContext.AddAsync(order);
                    await EshopDbContext.SaveChangesAsync();



                    HttpContext.Session.Remove(orderItemsString);
                    HttpContext.Session.Remove(totalPriceString);

                    return RedirectToAction(nameof(CustomerOrdersController.Index), nameof(CustomerOrdersController).Replace("Controller", ""), new { Area = nameof(Customer) });

                }
            }

            return RedirectToAction(nameof(HomeController.Index), nameof(HomeController).Replace("Controller", ""), new { Area = "" });

        }
    }
}
