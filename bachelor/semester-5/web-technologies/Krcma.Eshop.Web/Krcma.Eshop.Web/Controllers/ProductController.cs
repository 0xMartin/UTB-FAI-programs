using Krcma.Eshop.Web.Models.Database;
using Krcma.Eshop.Web.Models.Entity;
using Krcma.Eshop.Web.Models.ViewModels;
using Microsoft.AspNetCore.Mvc;
using System;
using System.Linq;

namespace Krcma.Eshop.Web.Controllers
{
    public class ProductController : Controller
    {
        readonly EshopDbContext eshopDbContext;

        public ProductController(EshopDbContext eshopDb)
        {
            eshopDbContext = eshopDb;
        }
        public IActionResult Detail(int ID)
        {
            Product product = eshopDbContext.Products.FirstOrDefault(p => p.ID == ID);
            if (product != null)
            {
                return View(product);
            }
            else
            {
                return NotFound();
            }
        }

        public IActionResult Shop(int PAGE)
        {
            ProductsViewModel pvm = new ProductsViewModel
            {
                Products = eshopDbContext.Products.ToArray(),
                ItemsOnPage = 12,
                CurrentPageNumber = PAGE
            };

            pvm.TotalPageCount = (int)Math.Ceiling(pvm.Products.Count / (float)pvm.ItemsOnPage);
            if (pvm.CurrentPageNumber < 1 || pvm.CurrentPageNumber > pvm.TotalPageCount)
            {
                return NotFound();
            } else
            {
                return View(pvm);
            }
        }

    }
}
