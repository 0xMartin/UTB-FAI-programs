using Krcma.Eshop.Web.Areas.ReturnModule;
using Krcma.Eshop.Web.Areas.ReturnModule.Entity;
using Krcma.Eshop.Web.Models.Entity;
using Krcma.Eshop.Web.Models.Identity;
using Microsoft.AspNetCore.Identity;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace Krcma.Eshop.Web.Models.Database
{
    public class DatabaseInit
    {
        public void Initialization(EshopDbContext eshopDbContext)
        {
            eshopDbContext.Database.EnsureCreated();

            if (eshopDbContext.CarouselItems.Count() == 0)
            {
                IList<CarouselItem> items = GenerateCarouselItems();
                foreach (var item in items)
                {
                    eshopDbContext.CarouselItems.Add(item);
                }

                eshopDbContext.SaveChanges();
            }

            if (eshopDbContext.Products.Count() == 0)
            {
                IList<Product> items = GenerateProducts();
                foreach (var item in items)
                {
                    eshopDbContext.Products.Add(item);
                }

                eshopDbContext.SaveChanges();
            }

            if(eshopDbContext.ReturnProperties.Count() == 0)
            {
                List<Property> propts = ReturnModuleShared.DefaultPropts();
                foreach (var item in propts)
                {
                    eshopDbContext.ReturnProperties.Add(item);
                }
                eshopDbContext.SaveChanges();
            }
        }

        public List<CarouselItem> GenerateCarouselItems()
        {
            List<CarouselItem> carouselItems = new List<CarouselItem>();

            carouselItems.Add(new CarouselItem()
            {
                ID = 0,
                ImageSource = "/img/Carousels/img1.png",
                ImageAlt = "Image 1"
            });

            carouselItems.Add(new CarouselItem()
            {
                ID = 2,
                ImageSource = "/img/Carousels/img2.png",
                ImageAlt = "Image 2"
            });

            carouselItems.Add(new CarouselItem()
            {
                ID = 3,
                ImageSource = "/img/Carousels/img3.png",
                ImageAlt = "Image 3"
            });

            return carouselItems;
        }
        public List<Product> GenerateProducts()
        {
            List<Product> products = new List<Product>();

            products.Add(new Product()
            {
                ID = 0,
                Name = "Apple AirPods Pro 2019",
                ImageSource = "/img/Products/img1.jpg",
                Description = "Bezdrátová sluchátka s mikrofonem, True Wireless špunty, uzavřená konstrukce, Bluetooth 5.0, aktivní potlačení hluku (ANC), hlasový asistent, přepínání skladeb, přijímání hovorů, certifikace IPX4, výdrž baterie až 24 h (4,5 h+19,5 h)",
                Price = 5790
            });

            products.Add(new Product()
            {
                ID = 1,
                Name = "Lenovo Legion 5 Pro 16ACH6H Storm Grey/Black kovový",
                ImageSource = "/img/Products/img2.jfif",
                Description = "Herní notebook - AMD Ryzen 7 5800H, 16\" IPS antireflexní 2560 × 1600 165Hz, RAM 16GB DDR4, NVIDIA GeForce RTX 3060 6GB 130 W, SSD 1000GB, numerická klávesnice, podsvícená RGB klávesnice, webkamera, USB - C, WiFi 6, Hmotnost 2.45 kg, Windows 10 Home",
                Price = 38567
            });

            products.Add(new Product()
            {
                ID = 2,
                Name = "65\" Samsung UE65TU7092",
                ImageSource = "/img/Products/img3.jfif",
                Description = "Televize SMART LED, 164cm, 4K Ultra HD, PQI 2000 (50Hz), HDR10, HDR10+, HLG, DVB-T2/S2/C, H.265/HEVC, 2× HDMI, 1× USB, LAN, WiFi, DLNA, HbbTV 2.0, herní režim, Apple TV, O2 TV, Netflix, HBO GO, Steam Link, Voyo, Apple Airplay 2, Tizen, repro 20W, Dolby Digital+, G",
                Price = 17990
            });

            return products;
        }

        public async Task EnsureRoleCreated(RoleManager<Role> roleManager)
        {
            string[] roles = Enum.GetNames(typeof(Roles));

            foreach (var role in roles)
            {
                await roleManager.CreateAsync(new Role(role));
            }
        }

        public async Task EnsureAdminCreated(UserManager<User> userManager)
        {
            User user = new User
            {
                UserName = "admin",
                Email = "admin@admin.cz",
                EmailConfirmed = true,
                FirstName = "Martin",
                LastName = "Krcma"
            };
            string password = "abc";

            User adminInDatabase = await userManager.FindByNameAsync(user.UserName);

            if (adminInDatabase == null)
            {

                IdentityResult result = await userManager.CreateAsync(user, password);

                if (result == IdentityResult.Success)
                {
                    string[] roles = Enum.GetNames(typeof(Roles));
                    foreach (var role in roles)
                    {
                        await userManager.AddToRoleAsync(user, role);
                    }
                }
                else if (result != null && result.Errors != null && result.Errors.Count() > 0)
                {
                    foreach (var error in result.Errors)
                    {
                        Debug.WriteLine($"Error during Role creation for Admin: {error.Code}, {error.Description}");
                    }
                }
            }

        }

        public async Task EnsureManagerCreated(UserManager<User> userManager)
        {
            User user = new User
            {
                UserName = "manager",
                Email = "manager@manager.cz",
                EmailConfirmed = true,
                FirstName = "Jan",
                LastName = "Novak"
            };
            string password = "abc";

            User managerInDatabase = await userManager.FindByNameAsync(user.UserName);

            if (managerInDatabase == null)
            {

                IdentityResult result = await userManager.CreateAsync(user, password);

                if (result == IdentityResult.Success)
                {
                    string[] roles = Enum.GetNames(typeof(Roles));
                    foreach (var role in roles)
                    {
                        if (role != Roles.Admin.ToString())
                            await userManager.AddToRoleAsync(user, role);
                    }
                }
                else if (result != null && result.Errors != null && result.Errors.Count() > 0)
                {
                    foreach (var error in result.Errors)
                    {
                        Debug.WriteLine($"Error during Role creation for Manager: {error.Code}, {error.Description}");
                    }
                }
            }

        }
    }
}
