#pragma checksum "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml" "{ff1816ec-aa5e-4d10-87f7-6f4963833460}" "55d59e8d563d58bd3b04135d2ee0f70e37c7fcf4"
// <auto-generated/>
#pragma warning disable 1591
[assembly: global::Microsoft.AspNetCore.Razor.Hosting.RazorCompiledItemAttribute(typeof(AspNetCore.Areas_Admin_Views_Users_Index), @"mvc.1.0.view", @"/Areas/Admin/Views/Users/Index.cshtml")]
namespace AspNetCore
{
    #line hidden
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading.Tasks;
    using Microsoft.AspNetCore.Mvc;
    using Microsoft.AspNetCore.Mvc.Rendering;
    using Microsoft.AspNetCore.Mvc.ViewFeatures;
#nullable restore
#line 1 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\_ViewImports.cshtml"
using Krcma.Eshop.Web;

#line default
#line hidden
#nullable disable
#nullable restore
#line 2 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\_ViewImports.cshtml"
using Krcma.Eshop.Web.Models;

#line default
#line hidden
#nullable disable
#nullable restore
#line 3 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\_ViewImports.cshtml"
using Krcma.Eshop.Web.Models.Identity;

#line default
#line hidden
#nullable disable
#nullable restore
#line 4 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\_ViewImports.cshtml"
using Krcma.Eshop.Web.Models.Entity;

#line default
#line hidden
#nullable disable
#nullable restore
#line 5 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\_ViewImports.cshtml"
using Krcma.Eshop.Web.Models.ViewModels;

#line default
#line hidden
#nullable disable
#nullable restore
#line 6 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\_ViewImports.cshtml"
using Krcma.Eshop.Web.Areas.Admin.Controllers;

#line default
#line hidden
#nullable disable
    [global::Microsoft.AspNetCore.Razor.Hosting.RazorSourceChecksumAttribute(@"SHA1", @"55d59e8d563d58bd3b04135d2ee0f70e37c7fcf4", @"/Areas/Admin/Views/Users/Index.cshtml")]
    [global::Microsoft.AspNetCore.Razor.Hosting.RazorSourceChecksumAttribute(@"SHA1", @"675885ad93ec9a6c484497bc54793f0864cffd66", @"/Areas/Admin/Views/_ViewImports.cshtml")]
    public class Areas_Admin_Views_Users_Index : global::Microsoft.AspNetCore.Mvc.Razor.RazorPage<IEnumerable<User>>
    {
        private static readonly global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute __tagHelperAttribute_0 = new global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute("rel", new global::Microsoft.AspNetCore.Html.HtmlString("stylesheet"), global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
        private static readonly global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute __tagHelperAttribute_1 = new global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute("href", new global::Microsoft.AspNetCore.Html.HtmlString("~/css/homepage-styles.css"), global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
        private static readonly global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute __tagHelperAttribute_2 = new global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute("asp-action", "Create", global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
        private static readonly global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute __tagHelperAttribute_3 = new global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute("asp-action", "Edit", global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
        private static readonly global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute __tagHelperAttribute_4 = new global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute("asp-action", "Details", global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
        private static readonly global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute __tagHelperAttribute_5 = new global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute("asp-action", "Delete", global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
        #line hidden
        #pragma warning disable 0649
        private global::Microsoft.AspNetCore.Razor.Runtime.TagHelpers.TagHelperExecutionContext __tagHelperExecutionContext;
        #pragma warning restore 0649
        private global::Microsoft.AspNetCore.Razor.Runtime.TagHelpers.TagHelperRunner __tagHelperRunner = new global::Microsoft.AspNetCore.Razor.Runtime.TagHelpers.TagHelperRunner();
        #pragma warning disable 0169
        private string __tagHelperStringValueBuffer;
        #pragma warning restore 0169
        private global::Microsoft.AspNetCore.Razor.Runtime.TagHelpers.TagHelperScopeManager __backed__tagHelperScopeManager = null;
        private global::Microsoft.AspNetCore.Razor.Runtime.TagHelpers.TagHelperScopeManager __tagHelperScopeManager
        {
            get
            {
                if (__backed__tagHelperScopeManager == null)
                {
                    __backed__tagHelperScopeManager = new global::Microsoft.AspNetCore.Razor.Runtime.TagHelpers.TagHelperScopeManager(StartTagHelperWritingScope, EndTagHelperWritingScope);
                }
                return __backed__tagHelperScopeManager;
            }
        }
        private global::Microsoft.AspNetCore.Mvc.Razor.TagHelpers.UrlResolutionTagHelper __Microsoft_AspNetCore_Mvc_Razor_TagHelpers_UrlResolutionTagHelper;
        private global::Microsoft.AspNetCore.Mvc.TagHelpers.AnchorTagHelper __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper;
        #pragma warning disable 1998
        public async override global::System.Threading.Tasks.Task ExecuteAsync()
        {
            WriteLiteral("\n");
#nullable restore
#line 3 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
   ViewData["Title"] = "Index"; 

#line default
#line hidden
#nullable disable
            WriteLiteral("\n");
            DefineSection("Styles", async() => {
                WriteLiteral("\n    ");
                __tagHelperExecutionContext = __tagHelperScopeManager.Begin("link", global::Microsoft.AspNetCore.Razor.TagHelpers.TagMode.StartTagOnly, "55d59e8d563d58bd3b04135d2ee0f70e37c7fcf46538", async() => {
                }
                );
                __Microsoft_AspNetCore_Mvc_Razor_TagHelpers_UrlResolutionTagHelper = CreateTagHelper<global::Microsoft.AspNetCore.Mvc.Razor.TagHelpers.UrlResolutionTagHelper>();
                __tagHelperExecutionContext.Add(__Microsoft_AspNetCore_Mvc_Razor_TagHelpers_UrlResolutionTagHelper);
                __tagHelperExecutionContext.AddHtmlAttribute(__tagHelperAttribute_0);
                __tagHelperExecutionContext.AddHtmlAttribute(__tagHelperAttribute_1);
                await __tagHelperRunner.RunAsync(__tagHelperExecutionContext);
                if (!__tagHelperExecutionContext.Output.IsContentModified)
                {
                    await __tagHelperExecutionContext.SetOutputContentAsync();
                }
                Write(__tagHelperExecutionContext.Output);
                __tagHelperExecutionContext = __tagHelperScopeManager.End();
                WriteLiteral("\n");
            }
            );
            WriteLiteral("\n<h1>Index</h1>\n\n<p>\n    ");
            __tagHelperExecutionContext = __tagHelperScopeManager.Begin("a", global::Microsoft.AspNetCore.Razor.TagHelpers.TagMode.StartTagAndEndTag, "55d59e8d563d58bd3b04135d2ee0f70e37c7fcf47798", async() => {
                WriteLiteral("Create New");
            }
            );
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper = CreateTagHelper<global::Microsoft.AspNetCore.Mvc.TagHelpers.AnchorTagHelper>();
            __tagHelperExecutionContext.Add(__Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper);
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.Action = (string)__tagHelperAttribute_2.Value;
            __tagHelperExecutionContext.AddTagHelperAttribute(__tagHelperAttribute_2);
            await __tagHelperRunner.RunAsync(__tagHelperExecutionContext);
            if (!__tagHelperExecutionContext.Output.IsContentModified)
            {
                await __tagHelperExecutionContext.SetOutputContentAsync();
            }
            Write(__tagHelperExecutionContext.Output);
            __tagHelperExecutionContext = __tagHelperScopeManager.End();
            WriteLiteral("\n</p>\n<table class=\"table table-responsive\">\n");
            WriteLiteral("    <thead>\n        <tr>\n            <th>\n                ");
#nullable restore
#line 20 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
           Write(Html.DisplayNameFor(model => model.FirstName));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n            </th>\n            <th>\n                ");
#nullable restore
#line 23 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
           Write(Html.DisplayNameFor(model => model.LastName));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n            </th>\n            <th>\n                ");
#nullable restore
#line 26 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
           Write(Html.DisplayNameFor(model => model.UserName));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n            </th>\n            <th>\n                ");
#nullable restore
#line 29 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
           Write(Html.DisplayNameFor(model => model.NormalizedUserName));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n            </th>\n            <th>\n                ");
#nullable restore
#line 32 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
           Write(Html.DisplayNameFor(model => model.Email));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n            </th>\n            <th>\n                ");
#nullable restore
#line 35 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
           Write(Html.DisplayNameFor(model => model.NormalizedEmail));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n            </th>\n            <th>\n                ");
#nullable restore
#line 38 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
           Write(Html.DisplayNameFor(model => model.EmailConfirmed));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n            </th>\n            <th>\n                ");
#nullable restore
#line 41 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
           Write(Html.DisplayNameFor(model => model.PasswordHash));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n            </th>\n            <th>\n                ");
#nullable restore
#line 44 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
           Write(Html.DisplayNameFor(model => model.SecurityStamp));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n            </th>\n            <th>\n                ");
#nullable restore
#line 47 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
           Write(Html.DisplayNameFor(model => model.ConcurrencyStamp));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n            </th>\n            <th>\n                ");
#nullable restore
#line 50 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
           Write(Html.DisplayNameFor(model => model.PhoneNumber));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n            </th>\n            <th>\n                ");
#nullable restore
#line 53 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
           Write(Html.DisplayNameFor(model => model.PhoneNumberConfirmed));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n            </th>\n            <th>\n                ");
#nullable restore
#line 56 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
           Write(Html.DisplayNameFor(model => model.TwoFactorEnabled));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n            </th>\n            <th>\n                ");
#nullable restore
#line 59 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
           Write(Html.DisplayNameFor(model => model.LockoutEnd));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n            </th>\n            <th>\n                ");
#nullable restore
#line 62 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
           Write(Html.DisplayNameFor(model => model.LockoutEnabled));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n            </th>\n            <th>\n                ");
#nullable restore
#line 65 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
           Write(Html.DisplayNameFor(model => model.AccessFailedCount));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n            </th>\n            <th></th>\n        </tr>\n    </thead>\n    <tbody>\n");
#nullable restore
#line 71 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
         foreach (var item in Model)
        {

#line default
#line hidden
#nullable disable
            WriteLiteral("<tr>\n    <td>\n        ");
#nullable restore
#line 75 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
   Write(Html.DisplayFor(modelItem => item.FirstName));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n    </td>\n    <td>\n        ");
#nullable restore
#line 78 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
   Write(Html.DisplayFor(modelItem => item.LastName));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n    </td>\n    <td>\n        ");
#nullable restore
#line 81 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
   Write(Html.DisplayFor(modelItem => item.UserName));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n    </td>\n    <td>\n        ");
#nullable restore
#line 84 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
   Write(Html.DisplayFor(modelItem => item.NormalizedUserName));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n    </td>\n    <td>\n        ");
#nullable restore
#line 87 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
   Write(Html.DisplayFor(modelItem => item.Email));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n    </td>\n    <td>\n        ");
#nullable restore
#line 90 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
   Write(Html.DisplayFor(modelItem => item.NormalizedEmail));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n    </td>\n    <td>\n        ");
#nullable restore
#line 93 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
   Write(Html.DisplayFor(modelItem => item.EmailConfirmed));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n    </td>\n    <td>\n        ");
#nullable restore
#line 96 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
   Write(Html.DisplayFor(modelItem => item.PasswordHash));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n    </td>\n    <td>\n        ");
#nullable restore
#line 99 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
   Write(Html.DisplayFor(modelItem => item.SecurityStamp));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n    </td>\n    <td>\n        ");
#nullable restore
#line 102 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
   Write(Html.DisplayFor(modelItem => item.ConcurrencyStamp));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n    </td>\n    <td>\n        ");
#nullable restore
#line 105 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
   Write(Html.DisplayFor(modelItem => item.PhoneNumber));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n    </td>\n    <td>\n        ");
#nullable restore
#line 108 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
   Write(Html.DisplayFor(modelItem => item.PhoneNumberConfirmed));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n    </td>\n    <td>\n        ");
#nullable restore
#line 111 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
   Write(Html.DisplayFor(modelItem => item.TwoFactorEnabled));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n    </td>\n    <td>\n        ");
#nullable restore
#line 114 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
   Write(Html.DisplayFor(modelItem => item.LockoutEnd));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n    </td>\n    <td>\n        ");
#nullable restore
#line 117 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
   Write(Html.DisplayFor(modelItem => item.LockoutEnabled));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n    </td>\n    <td>\n        ");
#nullable restore
#line 120 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
   Write(Html.DisplayFor(modelItem => item.AccessFailedCount));

#line default
#line hidden
#nullable disable
            WriteLiteral("\n    </td>\n    <td>\n        ");
            __tagHelperExecutionContext = __tagHelperScopeManager.Begin("a", global::Microsoft.AspNetCore.Razor.TagHelpers.TagMode.StartTagAndEndTag, "55d59e8d563d58bd3b04135d2ee0f70e37c7fcf419767", async() => {
                WriteLiteral("Edit");
            }
            );
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper = CreateTagHelper<global::Microsoft.AspNetCore.Mvc.TagHelpers.AnchorTagHelper>();
            __tagHelperExecutionContext.Add(__Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper);
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.Action = (string)__tagHelperAttribute_3.Value;
            __tagHelperExecutionContext.AddTagHelperAttribute(__tagHelperAttribute_3);
            if (__Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.RouteValues == null)
            {
                throw new InvalidOperationException(InvalidTagHelperIndexerAssignment("asp-route-id", "Microsoft.AspNetCore.Mvc.TagHelpers.AnchorTagHelper", "RouteValues"));
            }
            BeginWriteTagHelperAttribute();
#nullable restore
#line 123 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
                               WriteLiteral(item.Id);

#line default
#line hidden
#nullable disable
            __tagHelperStringValueBuffer = EndWriteTagHelperAttribute();
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.RouteValues["id"] = __tagHelperStringValueBuffer;
            __tagHelperExecutionContext.AddTagHelperAttribute("asp-route-id", __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.RouteValues["id"], global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
            await __tagHelperRunner.RunAsync(__tagHelperExecutionContext);
            if (!__tagHelperExecutionContext.Output.IsContentModified)
            {
                await __tagHelperExecutionContext.SetOutputContentAsync();
            }
            Write(__tagHelperExecutionContext.Output);
            __tagHelperExecutionContext = __tagHelperScopeManager.End();
            WriteLiteral(" |\n        ");
            __tagHelperExecutionContext = __tagHelperScopeManager.Begin("a", global::Microsoft.AspNetCore.Razor.TagHelpers.TagMode.StartTagAndEndTag, "55d59e8d563d58bd3b04135d2ee0f70e37c7fcf421939", async() => {
                WriteLiteral("Details");
            }
            );
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper = CreateTagHelper<global::Microsoft.AspNetCore.Mvc.TagHelpers.AnchorTagHelper>();
            __tagHelperExecutionContext.Add(__Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper);
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.Action = (string)__tagHelperAttribute_4.Value;
            __tagHelperExecutionContext.AddTagHelperAttribute(__tagHelperAttribute_4);
            if (__Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.RouteValues == null)
            {
                throw new InvalidOperationException(InvalidTagHelperIndexerAssignment("asp-route-id", "Microsoft.AspNetCore.Mvc.TagHelpers.AnchorTagHelper", "RouteValues"));
            }
            BeginWriteTagHelperAttribute();
#nullable restore
#line 124 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
                                  WriteLiteral(item.Id);

#line default
#line hidden
#nullable disable
            __tagHelperStringValueBuffer = EndWriteTagHelperAttribute();
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.RouteValues["id"] = __tagHelperStringValueBuffer;
            __tagHelperExecutionContext.AddTagHelperAttribute("asp-route-id", __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.RouteValues["id"], global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
            await __tagHelperRunner.RunAsync(__tagHelperExecutionContext);
            if (!__tagHelperExecutionContext.Output.IsContentModified)
            {
                await __tagHelperExecutionContext.SetOutputContentAsync();
            }
            Write(__tagHelperExecutionContext.Output);
            __tagHelperExecutionContext = __tagHelperScopeManager.End();
            WriteLiteral(" |\n        ");
            __tagHelperExecutionContext = __tagHelperScopeManager.Begin("a", global::Microsoft.AspNetCore.Razor.TagHelpers.TagMode.StartTagAndEndTag, "55d59e8d563d58bd3b04135d2ee0f70e37c7fcf424117", async() => {
                WriteLiteral("Delete");
            }
            );
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper = CreateTagHelper<global::Microsoft.AspNetCore.Mvc.TagHelpers.AnchorTagHelper>();
            __tagHelperExecutionContext.Add(__Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper);
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.Action = (string)__tagHelperAttribute_5.Value;
            __tagHelperExecutionContext.AddTagHelperAttribute(__tagHelperAttribute_5);
            if (__Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.RouteValues == null)
            {
                throw new InvalidOperationException(InvalidTagHelperIndexerAssignment("asp-route-id", "Microsoft.AspNetCore.Mvc.TagHelpers.AnchorTagHelper", "RouteValues"));
            }
            BeginWriteTagHelperAttribute();
#nullable restore
#line 125 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
                                 WriteLiteral(item.Id);

#line default
#line hidden
#nullable disable
            __tagHelperStringValueBuffer = EndWriteTagHelperAttribute();
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.RouteValues["id"] = __tagHelperStringValueBuffer;
            __tagHelperExecutionContext.AddTagHelperAttribute("asp-route-id", __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.RouteValues["id"], global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
            await __tagHelperRunner.RunAsync(__tagHelperExecutionContext);
            if (!__tagHelperExecutionContext.Output.IsContentModified)
            {
                await __tagHelperExecutionContext.SetOutputContentAsync();
            }
            Write(__tagHelperExecutionContext.Output);
            __tagHelperExecutionContext = __tagHelperScopeManager.End();
            WriteLiteral("\n    </td>\n</tr>");
#nullable restore
#line 127 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\Admin\Views\Users\Index.cshtml"
     }

#line default
#line hidden
#nullable disable
            WriteLiteral("    </tbody>\n</table>\n");
        }
        #pragma warning restore 1998
        [global::Microsoft.AspNetCore.Mvc.Razor.Internal.RazorInjectAttribute]
        public global::Microsoft.AspNetCore.Mvc.ViewFeatures.IModelExpressionProvider ModelExpressionProvider { get; private set; }
        [global::Microsoft.AspNetCore.Mvc.Razor.Internal.RazorInjectAttribute]
        public global::Microsoft.AspNetCore.Mvc.IUrlHelper Url { get; private set; }
        [global::Microsoft.AspNetCore.Mvc.Razor.Internal.RazorInjectAttribute]
        public global::Microsoft.AspNetCore.Mvc.IViewComponentHelper Component { get; private set; }
        [global::Microsoft.AspNetCore.Mvc.Razor.Internal.RazorInjectAttribute]
        public global::Microsoft.AspNetCore.Mvc.Rendering.IJsonHelper Json { get; private set; }
        [global::Microsoft.AspNetCore.Mvc.Razor.Internal.RazorInjectAttribute]
        public global::Microsoft.AspNetCore.Mvc.Rendering.IHtmlHelper<IEnumerable<User>> Html { get; private set; }
    }
}
#pragma warning restore 1591
