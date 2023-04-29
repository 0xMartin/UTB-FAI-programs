using Krcma.Eshop.Web.Models;
using Krcma.Eshop.Web.Models.Database;
using Krcma.Eshop.Web.Models.ViewModels;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using System;
using System.Diagnostics;
using System.Linq;

namespace Krcma.Eshop.Web.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;
        readonly EshopDbContext eshopDbContext;

        public HomeController(ILogger<HomeController> logger, EshopDbContext eshopDb)
        {
            _logger = logger;
            eshopDbContext = eshopDb;
        }

        public IActionResult Index()
        {
            _logger.LogInformation("Loaded Home Index");


            IndexViewModel indexVM = new IndexViewModel();
            indexVM.Products = eshopDbContext.Products.ToList();

            Random rng = new Random();
            indexVM.Products = indexVM.Products.OrderBy(x => rng.Next()).ToArray();

            indexVM.CarouselItems = eshopDbContext.CarouselItems.ToList();

            return View(indexVM);
        }

        public IActionResult Privacy()
        {
            return View();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}
