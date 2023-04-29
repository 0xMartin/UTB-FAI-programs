using System;
using System.Linq;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

using Xunit;
using Xunit.Abstractions;
using Moq;

using Krcma.Eshop.Web.Models.Entity;
using Krcma.Eshop.Web.Models.Database;
using Krcma.Eshop.Web.Areas.Admin.Controllers;
using Vogeltanz.Eshop.Tests.Helpers;

namespace Krcma.Eshop.Tests
{
    public class CarouselControllerTests
    {
        const string relativeCarouselDirectoryPath = "/img/Carousels";

        private readonly ITestOutputHelper _testOutputHelper;
        public CarouselControllerTests(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        [Fact]
        public async Task CarouselCreate_ValidSuccess()
        {
            // Arrange
            var mockIWebHostEnvironment = new Mock<IWebHostEnvironment>();
            mockIWebHostEnvironment.Setup(webHostEnv => webHostEnv.WebRootPath).Returns(Directory.GetCurrentDirectory());

            //Nainstalován Nuget package: Microsoft.EntityFrameworkCore.InMemory
            //databazi vytvori v pameti
            //Jsou zde konkretni tridy, takze to neni uplne OK - mely by se vyuzit interface jako treba pres IUnitOfWork, IRepository<T>, nebo pres vlastni IDbContext (je pak ale nutne vyuzivat interface i v hlavnim projektu, jinak v unit testech nebude spravne fungovat mockovani)
            //takto to ale v krizovych situacich taky jde :-)
            DbContextOptions options = new DbContextOptionsBuilder<EshopDbContext>()
                                       .UseInMemoryDatabase(databaseName: Guid.NewGuid().ToString())
                                       .Options;
            var databaseContext = new EshopDbContext(options);
            databaseContext.Database.EnsureCreated();


            CarouselController controller = new CarouselController(databaseContext, mockIWebHostEnvironment.Object);
            controller.ObjectValidator = new ObjectValidator();
            IActionResult iActionResult = null;



            string content = "‰PNG" + "FakeImageContent";
            string fileName = "UploadImageFile.png";

            Directory.CreateDirectory(Path.Combine(Directory.GetCurrentDirectory() + relativeCarouselDirectoryPath));

            //nastavení fakeové IFormFile pomocí MemoryStream
            using (var ms = new MemoryStream())
            {
                using (var writer = new StreamWriter(ms))
                {
                    IFormFileMockHelper iffMockHelper = new IFormFileMockHelper(_testOutputHelper);
                    Mock<IFormFile> iffMock = iffMockHelper.MockIFormFile(ms, writer, fileName, content, "image/png");
                    CarouselItem testCarousel = GetTestCarouselItem(iffMock.Object);

                    //Act
                    iActionResult = await controller.Create(testCarousel);

                }
            }

            // Assert
            RedirectToActionResult redirect = Assert.IsType<RedirectToActionResult>(iActionResult);
            Assert.Matches(redirect.ActionName, nameof(CarouselController.Select));

            /*var viewResult = Assert.IsType<ViewResult>(iActionResult);
            var model = Assert.IsAssignableFrom<IList<CarouselItem>>(viewResult.ViewData.Model);
            */

            int carouselCount = (await databaseContext.CarouselItems.ToListAsync()).Count;
            Assert.Equal(1, carouselCount);

            Assert.Single(await databaseContext.CarouselItems.ToListAsync());

        }


        /*[Fact]
        public async Task CarouselCreate_ValidFailure()
        {
            
        }*/


        [Fact]
        public async Task CarouselCreate_InvalidFailure()
        {
            //zkuste doplnit tak, aby objekt pro CarouselItem nebyl validní, což bude správný výsledek testu ;-)
            
            //pozn. to aby test neprosel je v tuto chvili napsano schvalne, aby vas to upozornilo, ze mate jeste neco udelat
            Assert.True(false);
        }



        CarouselItem GetTestCarouselItem(IFormFile iff)
        {
            return new CarouselItem()
            {
                ImageSource = null,
                ImageAlt = "image",
                Image = iff
            };
        }

    }
}
