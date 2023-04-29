using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

using Microsoft.AspNetCore.Mvc;

using Xunit;
using Moq;

using Krcma.Eshop.Web.Areas.Security.Controllers;
using Krcma.Eshop.Web.Controllers;
using Krcma.Eshop.Web.Models.ApplicationServices.Abstraction;
using Krcma.Eshop.Web.Models.ViewModels;

namespace Krcma.Eshop.Tests
{
    public class AccountControllerTests
    {
        [Fact]
        public async Task Login_ValidSuccess()
        {
            // Arrange
            var mockISecurityApplicationService = new Mock<ISecurityApplicationService>();
            mockISecurityApplicationService.Setup(security => security.Login(It.IsAny<LoginViewModel>()))
                                                                      //první verze, kdy prostě řekneme, že login projde a hotovo :-)
                                                                      .Returns(() => Task<bool>.Run(() => true));
                                                                      //druhá verze, kdy si můžeme testovat, co je v LoginViewModel:
                                                                      //.Returns<LoginViewModel>((loginVM) => {return Task<bool>.Run(() =>
                                                                      //{
                                                                      //    if (loginVM.Username == "superadmin" && loginVM.Password == "123")
                                                                      //    { return true; }
                                                                      //    else
                                                                      //    { return false; }
                                                                      //});});


            LoginViewModel loginViewModel = new LoginViewModel()
            {
                Username = "superadmin",
                Password = "123"
            };


            AccountController controller = new AccountController(mockISecurityApplicationService.Object);
            //pokud chci vypnout validaci, tak nenastavuju ObjectValidator
            //(je to na vás, jak to u Unit Testů uděláte, ale pokud v controlleru používáte TryValidateModel(model), tak jej nějak nastavit musíte ... stejně tak pokud chcete testovat případ, kdy objekt není validní)
            //controller.ObjectValidator = new ObjectValidator();
            IActionResult iActionResult = null;


            //Act
            iActionResult = await controller.Login(loginViewModel);


            // Assert
            RedirectToActionResult redirect = Assert.IsType<RedirectToActionResult>(iActionResult);
            Assert.Matches(redirect.ActionName, nameof(HomeController.Index));
            Assert.Matches(redirect.ControllerName, nameof(HomeController).Replace("Controller", String.Empty));
            Assert.Matches(redirect.RouteValues.Single((pair) => pair.Key == "area").Value.ToString(), String.Empty);


        }
    }
}
