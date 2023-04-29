using Krcma.Eshop.Web.Models.Database;
using Krcma.Eshop.Web.Models.Entity;
using Krcma.Eshop.Web.Models.Identity;
using Krcma.Eshop.Web.Models.Implementation;
using Krcma.Eshop.Web.Models.ViewModels;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc;
using System;
using System.Linq;
using System.Threading.Tasks;

namespace Krcma.Eshop.Web.Areas.Admin.Controllers
{
    [Area("Admin")]
    [Authorize(Roles = nameof(Roles.Admin) + ", " + nameof(Roles.Manager))]
    public class ProductController : Controller
    {
        readonly EshopDbContext eshopDbContext;
        IWebHostEnvironment env;

        public ProductController(EshopDbContext eshopDb, IWebHostEnvironment env)
        {
            this.eshopDbContext = eshopDb;
            this.env = env;
        }

        public IActionResult Select()
        {
            IndexViewModel indexVM = new IndexViewModel();
            indexVM.Products = eshopDbContext.Products.ToList();
            indexVM.CarouselItems = eshopDbContext.CarouselItems.ToList();
            return View(indexVM);
        }
        public IActionResult Create()
        {
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> Create(Product product)
        {
            if (String.IsNullOrWhiteSpace(product.Name) == false && product.Price != 0)
            {

                FileUpload fileUpload = new FileUpload(env.WebRootPath, "img/Products", "image");
                if (fileUpload.CheckFileContent(product.Image) && fileUpload.CheckFileLength(product.Image))
                {
                    product.ImageSource = await fileUpload.FileUploadAsync(product.Image);

                    ModelState.Clear();
                    TryValidateModel(product);

                    if (ModelState.IsValid)
                    {
                        eshopDbContext.Products.Add(product);

                        await eshopDbContext.SaveChangesAsync();

                        return RedirectToAction(nameof(ProductController.Select));
                    }
                }

                return View(product);
            }
            else
            {
                return View(product);
            }
        }
        public IActionResult Edit(int ID)
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

        [HttpPost]
        public async Task<IActionResult> Edit(Product p)
        {
            Product product = eshopDbContext.Products.FirstOrDefault(pr => pr.ID == p.ID);

            if (product != null)
            {

                if (product.Image != null)
                {
                    FileUpload fu = new FileUpload(env.WebRootPath, "img/Products", "image");
                    if (fu.CheckFileContent(product.Image) && fu.CheckFileLength(product.Image))
                    {
                        product.ImageSource = await fu.FileUploadAsync(product.Image);
                        if (String.IsNullOrWhiteSpace(product.ImageSource) == false)
                        {
                            product.ImageSource = product.ImageSource;
                        }
                    }
                }

                product.Name = p.Name;
                product.Price = p.Price;
                product.Description = p.Description;
                product.DigitalProduct = p.DigitalProduct;

                await eshopDbContext.SaveChangesAsync();

                return RedirectToAction(nameof(ProductController.Select));
            }
            else
            {
                return NotFound();
            }
        }
        public async Task<IActionResult> Delete(int ID)
        {
            Product p = eshopDbContext.Products.Where(p => p.ID == ID).FirstOrDefault();

            if (p != null)
            {
                eshopDbContext.Products.Remove(p);

                await eshopDbContext.SaveChangesAsync();
            }

            return RedirectToAction(nameof(ProductController.Select));
        }
    }
}
