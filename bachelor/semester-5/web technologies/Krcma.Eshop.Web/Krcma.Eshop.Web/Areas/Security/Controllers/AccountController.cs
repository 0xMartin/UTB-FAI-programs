using Krcma.Eshop.Web.Controllers;
using Krcma.Eshop.Web.Models.ApplicationServices.Abstraction;
using Krcma.Eshop.Web.Models.ViewModels;
using Microsoft.AspNetCore.Mvc;
using System;
using System.Threading.Tasks;

namespace Krcma.Eshop.Web.Areas.Security.Controllers
{
    [Area("Security")]
    public class AccountController : Controller
    {
        ISecurityApplicationService security;

        public AccountController(ISecurityApplicationService security)
        {
            this.security = security;
        }

        public IActionResult Register()
        {
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> Register(RegisterViewModel registerVm)
        {
            if (ModelState.IsValid)
            {
                string[] errors = await security.Register(registerVm, Models.Identity.Roles.Customer);

                if (errors != null)
                {
                    LoginViewModel loginVM = new LoginViewModel()
                    {
                        Username = registerVm.Username,
                        Password = registerVm.Password
                    };

                    bool isLogged = await security.Login(loginVM);
                    if (isLogged)
                    {
                        return RedirectToAction(nameof(HomeController.Index), nameof(HomeController).Replace("Controller", String.Empty), new { area = String.Empty });
                    }
                    else
                    {
                        return RedirectToAction(nameof(Login));
                    }
                }
            }

            return View(registerVm);
        }

        public IActionResult Login()
        {
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> Login(LoginViewModel loginVM)
        {
            if (ModelState.IsValid)
            {
                bool isLogged = await security.Login(loginVM);
                if (isLogged)
                {
                    return RedirectToAction(nameof(HomeController.Index), nameof(HomeController).Replace("Controller", String.Empty), new { area = String.Empty });
                }

                loginVM.LoginFail = true;

            }

            return View(loginVM);
        }

        public async Task<IActionResult> Logout(LoginViewModel loginVM)
        {
            await security.Logout();

            return RedirectToAction(nameof(Login));
        }

    }
}
