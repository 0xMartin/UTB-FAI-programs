#pragma checksum "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml" "{ff1816ec-aa5e-4d10-87f7-6f4963833460}" "f759975c72c1cc2ee40bb606d483486715474635"
// <auto-generated/>
#pragma warning disable 1591
[assembly: global::Microsoft.AspNetCore.Razor.Hosting.RazorCompiledItemAttribute(typeof(AspNetCore.Areas_ReturnModule_Views_Return_Index), @"mvc.1.0.view", @"/Areas/ReturnModule/Views/Return/Index.cshtml")]
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
#line 1 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\_ViewImports.cshtml"
using Krcma.Eshop.Web.Areas.ReturnModule.Models;

#line default
#line hidden
#nullable disable
#nullable restore
#line 2 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\_ViewImports.cshtml"
using Krcma.Eshop.Web.Areas.ReturnModule.Entity;

#line default
#line hidden
#nullable disable
#nullable restore
#line 3 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\_ViewImports.cshtml"
using Krcma.Eshop.Web.Models.Entity;

#line default
#line hidden
#nullable disable
    [global::Microsoft.AspNetCore.Razor.Hosting.RazorSourceChecksumAttribute(@"SHA1", @"f759975c72c1cc2ee40bb606d483486715474635", @"/Areas/ReturnModule/Views/Return/Index.cshtml")]
    [global::Microsoft.AspNetCore.Razor.Hosting.RazorSourceChecksumAttribute(@"SHA1", @"7ac61100559e15da5ef27640fde62e1fc4c182d7", @"/Areas/ReturnModule/Views/_ViewImports.cshtml")]
    public class Areas_ReturnModule_Views_Return_Index : global::Microsoft.AspNetCore.Mvc.Razor.RazorPage<ReturnViewModel>
    {
        private static readonly global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute __tagHelperAttribute_0 = new global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute("rel", new global::Microsoft.AspNetCore.Html.HtmlString("stylesheet"), global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
        private static readonly global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute __tagHelperAttribute_1 = new global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute("href", new global::Microsoft.AspNetCore.Html.HtmlString("~/css/homepage-styles.css"), global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
        private static readonly global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute __tagHelperAttribute_2 = new global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute("asp-area", "", global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
        private static readonly global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute __tagHelperAttribute_3 = new global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute("asp-controller", "Product", global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
        private static readonly global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute __tagHelperAttribute_4 = new global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute("asp-action", "Detail", global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
        private static readonly global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute __tagHelperAttribute_5 = new global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute("type", new global::Microsoft.AspNetCore.Html.HtmlString("button"), global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
        private static readonly global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute __tagHelperAttribute_6 = new global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute("class", new global::Microsoft.AspNetCore.Html.HtmlString("btn btn-danger"), global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
        private static readonly global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute __tagHelperAttribute_7 = new global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute("asp-area", "ReturnModule", global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
        private static readonly global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute __tagHelperAttribute_8 = new global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute("asp-controller", "Return", global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
        private static readonly global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute __tagHelperAttribute_9 = new global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute("asp-action", "RequestProductReturn", global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
        private static readonly global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute __tagHelperAttribute_10 = new global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute("onclick", new global::Microsoft.AspNetCore.Html.HtmlString("return ConfirmReturn();"), global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
        private static readonly global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute __tagHelperAttribute_11 = new global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute("class", new global::Microsoft.AspNetCore.Html.HtmlString("badge badge-dark p-3 m-1"), global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
        private static readonly global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute __tagHelperAttribute_12 = new global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute("asp-action", "Index", global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
        private static readonly global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute __tagHelperAttribute_13 = new global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute("class", new global::Microsoft.AspNetCore.Html.HtmlString("badge badge-secondary p-3 m-1"), global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
        private static readonly global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute __tagHelperAttribute_14 = new global::Microsoft.AspNetCore.Razor.TagHelpers.TagHelperAttribute("src", new global::Microsoft.AspNetCore.Html.HtmlString("~/js/returnModule.js"), global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
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
            WriteLiteral("\r\n");
#nullable restore
#line 3 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
  
    ViewData["Title"] = "Return product";

#line default
#line hidden
#nullable disable
            WriteLiteral("\r\n");
            DefineSection("Styles", async() => {
                WriteLiteral("\r\n    ");
                __tagHelperExecutionContext = __tagHelperScopeManager.Begin("link", global::Microsoft.AspNetCore.Razor.TagHelpers.TagMode.StartTagOnly, "f759975c72c1cc2ee40bb606d4834867154746359076", async() => {
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
                WriteLiteral("\r\n");
            }
            );
            WriteLiteral("\r\n<div class=\"px-5 pt-4\">\r\n    <h1>Return product</h1>\r\n\r\n    <div class=\"card p-2\">\r\n");
#nullable restore
#line 16 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
          
            if (Model != null && Model != null && Model.UserOrders.Count > 0)
            {

                int itemsOnPage = 0;
                int itemStartIndex = (Model.CurrentPageNumber - 1) * Model.ItemsOnPage;



#line default
#line hidden
#nullable disable
            WriteLiteral(@"                <table class=""table table-responsive table-striped w-100"" style=""border-collapse: separate; border-spacing: 0 10px;"">
                    <thead class=""thead-dark"">
                        <tr>
                            <th class=""col-sm-8 p-3"">Info</th>
                            <th class=""col-sm-4 p-3"">Image</th>
                        </tr>
                    </thead>
");
#nullable restore
#line 31 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
                      
                        foreach (var item in Model.UserOrders)
                        {
                            if ((itemsOnPage - itemStartIndex) > Model.ItemsOnPage) break;

                            foreach (var itemOrderItem in item.OrderItems)
                            {
                                itemsOnPage++;
                                if ((itemsOnPage - itemStartIndex) > Model.ItemsOnPage) break;
                                if (itemsOnPage <= itemStartIndex) continue;


#line default
#line hidden
#nullable disable
            WriteLiteral(@"                                <tr>
                                    <td class=""col-sm-8 p-0"">
                                        <table class=""table w-100"">
                                            <tr>
                                                <th>Product &#128230;</th>
                                                <td>
                                                    ");
            __tagHelperExecutionContext = __tagHelperScopeManager.Begin("a", global::Microsoft.AspNetCore.Razor.TagHelpers.TagMode.StartTagAndEndTag, "f759975c72c1cc2ee40bb606d48348671547463512455", async() => {
#nullable restore
#line 48 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
                                                                                                                                                    Write(itemOrderItem.Product.Name);

#line default
#line hidden
#nullable disable
            }
            );
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper = CreateTagHelper<global::Microsoft.AspNetCore.Mvc.TagHelpers.AnchorTagHelper>();
            __tagHelperExecutionContext.Add(__Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper);
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.Area = (string)__tagHelperAttribute_2.Value;
            __tagHelperExecutionContext.AddTagHelperAttribute(__tagHelperAttribute_2);
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.Controller = (string)__tagHelperAttribute_3.Value;
            __tagHelperExecutionContext.AddTagHelperAttribute(__tagHelperAttribute_3);
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.Action = (string)__tagHelperAttribute_4.Value;
            __tagHelperExecutionContext.AddTagHelperAttribute(__tagHelperAttribute_4);
            if (__Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.RouteValues == null)
            {
                throw new InvalidOperationException(InvalidTagHelperIndexerAssignment("asp-route-ID", "Microsoft.AspNetCore.Mvc.TagHelpers.AnchorTagHelper", "RouteValues"));
            }
            BeginWriteTagHelperAttribute();
#nullable restore
#line 48 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
                                                                                                                  WriteLiteral(itemOrderItem.Product.ID);

#line default
#line hidden
#nullable disable
            __tagHelperStringValueBuffer = EndWriteTagHelperAttribute();
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.RouteValues["ID"] = __tagHelperStringValueBuffer;
            __tagHelperExecutionContext.AddTagHelperAttribute("asp-route-ID", __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.RouteValues["ID"], global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
            await __tagHelperRunner.RunAsync(__tagHelperExecutionContext);
            if (!__tagHelperExecutionContext.Output.IsContentModified)
            {
                await __tagHelperExecutionContext.SetOutputContentAsync();
            }
            Write(__tagHelperExecutionContext.Output);
            __tagHelperExecutionContext = __tagHelperScopeManager.End();
            WriteLiteral(@"
                                                </td>
                                            </tr>
                                            <tr>
                                                <th>Price &#128178;</th>
                                                <td>");
#nullable restore
#line 53 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
                                               Write(itemOrderItem.Price.ToString("C2"));

#line default
#line hidden
#nullable disable
            WriteLiteral("</td>\r\n                                            </tr>\r\n                                            <tr>\r\n                                                <th>Amount &#x1F522;</th>\r\n                                                <td>");
#nullable restore
#line 57 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
                                               Write(itemOrderItem.Amount);

#line default
#line hidden
#nullable disable
            WriteLiteral("</td>\r\n                                            </tr>\r\n                                            <tr>\r\n                                                <th>Date Time Created &#128197;</th>\r\n                                                <td>");
#nullable restore
#line 61 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
                                               Write(item.DateTimeCreated);

#line default
#line hidden
#nullable disable
            WriteLiteral("</td>\r\n                                            </tr>\r\n                                            <tr>\r\n                                                <th>Order Number &#128230;</th>\r\n                                                <td>");
#nullable restore
#line 65 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
                                               Write(item.OrderNumber);

#line default
#line hidden
#nullable disable
            WriteLiteral("</td>\r\n                                            </tr>\r\n                                        </table>\r\n\r\n                                        <div class=\"px-4 py-2\">\r\n");
#nullable restore
#line 71 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
                                              
                                                ReturnProduct rp = Model.ReturnProducts.FirstOrDefault(rp => rp.OrderItemID == itemOrderItem.ID);
                                                if (rp != null)
                                                {
                                                    if (rp.Processed)
                                                    {
                                                        if (rp.Approved)
                                                        {

#line default
#line hidden
#nullable disable
            WriteLiteral("                                                            <p class=\"text-success\"><b>Return request approved</b></p>\r\n");
#nullable restore
#line 80 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
                                                        }
                                                        else
                                                        {

#line default
#line hidden
#nullable disable
            WriteLiteral("                                                            <p class=\"text-danger\"><b>Return request denied</b></p>\r\n");
#nullable restore
#line 84 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
                                                        }
                                                    }
                                                    else
                                                    {

#line default
#line hidden
#nullable disable
            WriteLiteral("                                                        <p class=\"text-info\"><b>Request sent, waiting for approval</b></p>\r\n");
#nullable restore
#line 89 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
                                                    }
                                                }
                                                else
                                                {

#line default
#line hidden
#nullable disable
            WriteLiteral("                                                    ");
            __tagHelperExecutionContext = __tagHelperScopeManager.Begin("a", global::Microsoft.AspNetCore.Razor.TagHelpers.TagMode.StartTagAndEndTag, "f759975c72c1cc2ee40bb606d48348671547463520398", async() => {
                WriteLiteral("Return Product");
            }
            );
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper = CreateTagHelper<global::Microsoft.AspNetCore.Mvc.TagHelpers.AnchorTagHelper>();
            __tagHelperExecutionContext.Add(__Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper);
            __tagHelperExecutionContext.AddHtmlAttribute(__tagHelperAttribute_5);
            __tagHelperExecutionContext.AddHtmlAttribute(__tagHelperAttribute_6);
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.Area = (string)__tagHelperAttribute_7.Value;
            __tagHelperExecutionContext.AddTagHelperAttribute(__tagHelperAttribute_7);
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.Controller = (string)__tagHelperAttribute_8.Value;
            __tagHelperExecutionContext.AddTagHelperAttribute(__tagHelperAttribute_8);
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.Action = (string)__tagHelperAttribute_9.Value;
            __tagHelperExecutionContext.AddTagHelperAttribute(__tagHelperAttribute_9);
            if (__Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.RouteValues == null)
            {
                throw new InvalidOperationException(InvalidTagHelperIndexerAssignment("asp-route-orderItemID", "Microsoft.AspNetCore.Mvc.TagHelpers.AnchorTagHelper", "RouteValues"));
            }
            BeginWriteTagHelperAttribute();
#nullable restore
#line 93 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
                                                                                                                                                                                         WriteLiteral(itemOrderItem.ID);

#line default
#line hidden
#nullable disable
            __tagHelperStringValueBuffer = EndWriteTagHelperAttribute();
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.RouteValues["orderItemID"] = __tagHelperStringValueBuffer;
            __tagHelperExecutionContext.AddTagHelperAttribute("asp-route-orderItemID", __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.RouteValues["orderItemID"], global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
            __tagHelperExecutionContext.AddHtmlAttribute(__tagHelperAttribute_10);
            await __tagHelperRunner.RunAsync(__tagHelperExecutionContext);
            if (!__tagHelperExecutionContext.Output.IsContentModified)
            {
                await __tagHelperExecutionContext.SetOutputContentAsync();
            }
            Write(__tagHelperExecutionContext.Output);
            __tagHelperExecutionContext = __tagHelperScopeManager.End();
            WriteLiteral("\r\n");
#nullable restore
#line 94 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
                                                }
                                            

#line default
#line hidden
#nullable disable
            WriteLiteral("                                        </div>\r\n\r\n                                    </td>\r\n                                    <td class=\"col-sm-4\"><img class=\"card-img-top p-2\"");
            BeginWriteAttribute("src", " src=\"", 5460, "\"", 5500, 1);
#nullable restore
#line 99 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
WriteAttributeValue("", 5466, itemOrderItem.Product.ImageSource, 5466, 34, false);

#line default
#line hidden
#nullable disable
            EndWriteAttribute();
            WriteLiteral(" /></td>\r\n                                </tr>\r\n");
#nullable restore
#line 101 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
                            }
                        }
                    

#line default
#line hidden
#nullable disable
            WriteLiteral("                </table>\r\n");
            WriteLiteral("                <div class=\"d-flex justify-content-center\">\r\n");
#nullable restore
#line 108 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
                      
                        for (int i = Math.Max(Model.CurrentPageNumber - 3, 1), cnt = 0; i <= Model.TotalPageCount && cnt <= 7; ++i, ++cnt)
                        {
                            if (i == Model.CurrentPageNumber)
                            {

#line default
#line hidden
#nullable disable
            WriteLiteral("                                ");
            __tagHelperExecutionContext = __tagHelperScopeManager.Begin("a", global::Microsoft.AspNetCore.Razor.TagHelpers.TagMode.StartTagAndEndTag, "f759975c72c1cc2ee40bb606d48348671547463525379", async() => {
#nullable restore
#line 113 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
                                                                                                                                                      Write(i);

#line default
#line hidden
#nullable disable
            }
            );
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper = CreateTagHelper<global::Microsoft.AspNetCore.Mvc.TagHelpers.AnchorTagHelper>();
            __tagHelperExecutionContext.Add(__Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper);
            __tagHelperExecutionContext.AddHtmlAttribute(__tagHelperAttribute_11);
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.Area = (string)__tagHelperAttribute_7.Value;
            __tagHelperExecutionContext.AddTagHelperAttribute(__tagHelperAttribute_7);
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.Controller = (string)__tagHelperAttribute_8.Value;
            __tagHelperExecutionContext.AddTagHelperAttribute(__tagHelperAttribute_8);
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.Action = (string)__tagHelperAttribute_12.Value;
            __tagHelperExecutionContext.AddTagHelperAttribute(__tagHelperAttribute_12);
            if (__Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.RouteValues == null)
            {
                throw new InvalidOperationException(InvalidTagHelperIndexerAssignment("asp-route-PAGE", "Microsoft.AspNetCore.Mvc.TagHelpers.AnchorTagHelper", "RouteValues"));
            }
            BeginWriteTagHelperAttribute();
#nullable restore
#line 113 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
                                                                                                                                           WriteLiteral(i);

#line default
#line hidden
#nullable disable
            __tagHelperStringValueBuffer = EndWriteTagHelperAttribute();
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.RouteValues["PAGE"] = __tagHelperStringValueBuffer;
            __tagHelperExecutionContext.AddTagHelperAttribute("asp-route-PAGE", __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.RouteValues["PAGE"], global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
            await __tagHelperRunner.RunAsync(__tagHelperExecutionContext);
            if (!__tagHelperExecutionContext.Output.IsContentModified)
            {
                await __tagHelperExecutionContext.SetOutputContentAsync();
            }
            Write(__tagHelperExecutionContext.Output);
            __tagHelperExecutionContext = __tagHelperScopeManager.End();
            WriteLiteral("\r\n");
#nullable restore
#line 114 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
                            }
                            else
                            {

#line default
#line hidden
#nullable disable
            WriteLiteral("                                ");
            __tagHelperExecutionContext = __tagHelperScopeManager.Begin("a", global::Microsoft.AspNetCore.Razor.TagHelpers.TagMode.StartTagAndEndTag, "f759975c72c1cc2ee40bb606d48348671547463528820", async() => {
#nullable restore
#line 117 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
                                                                                                                                                           Write(i);

#line default
#line hidden
#nullable disable
            }
            );
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper = CreateTagHelper<global::Microsoft.AspNetCore.Mvc.TagHelpers.AnchorTagHelper>();
            __tagHelperExecutionContext.Add(__Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper);
            __tagHelperExecutionContext.AddHtmlAttribute(__tagHelperAttribute_13);
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.Area = (string)__tagHelperAttribute_7.Value;
            __tagHelperExecutionContext.AddTagHelperAttribute(__tagHelperAttribute_7);
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.Controller = (string)__tagHelperAttribute_8.Value;
            __tagHelperExecutionContext.AddTagHelperAttribute(__tagHelperAttribute_8);
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.Action = (string)__tagHelperAttribute_12.Value;
            __tagHelperExecutionContext.AddTagHelperAttribute(__tagHelperAttribute_12);
            if (__Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.RouteValues == null)
            {
                throw new InvalidOperationException(InvalidTagHelperIndexerAssignment("asp-route-PAGE", "Microsoft.AspNetCore.Mvc.TagHelpers.AnchorTagHelper", "RouteValues"));
            }
            BeginWriteTagHelperAttribute();
#nullable restore
#line 117 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
                                                                                                                                                WriteLiteral(i);

#line default
#line hidden
#nullable disable
            __tagHelperStringValueBuffer = EndWriteTagHelperAttribute();
            __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.RouteValues["PAGE"] = __tagHelperStringValueBuffer;
            __tagHelperExecutionContext.AddTagHelperAttribute("asp-route-PAGE", __Microsoft_AspNetCore_Mvc_TagHelpers_AnchorTagHelper.RouteValues["PAGE"], global::Microsoft.AspNetCore.Razor.TagHelpers.HtmlAttributeValueStyle.DoubleQuotes);
            await __tagHelperRunner.RunAsync(__tagHelperExecutionContext);
            if (!__tagHelperExecutionContext.Output.IsContentModified)
            {
                await __tagHelperExecutionContext.SetOutputContentAsync();
            }
            Write(__tagHelperExecutionContext.Output);
            __tagHelperExecutionContext = __tagHelperScopeManager.End();
            WriteLiteral("\r\n");
#nullable restore
#line 118 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
                            }
                        }
                    

#line default
#line hidden
#nullable disable
            WriteLiteral("                </div>\r\n");
#nullable restore
#line 122 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
            }
            else
            {

#line default
#line hidden
#nullable disable
            WriteLiteral("                <h2>Orders are empty!</h2>\r\n");
#nullable restore
#line 126 "C:\Users\Krcma\OneDrive\Plocha\PW\Krcma.Eshop.Web\Krcma.Eshop.Web\Areas\ReturnModule\Views\Return\Index.cshtml"
            }
        

#line default
#line hidden
#nullable disable
            WriteLiteral("\r\n    </div>\r\n</div>\r\n\r\n");
            DefineSection("Scripts", async() => {
                WriteLiteral("\r\n    ");
                __tagHelperExecutionContext = __tagHelperScopeManager.Begin("script", global::Microsoft.AspNetCore.Razor.TagHelpers.TagMode.StartTagAndEndTag, "f759975c72c1cc2ee40bb606d48348671547463532946", async() => {
                }
                );
                __Microsoft_AspNetCore_Mvc_Razor_TagHelpers_UrlResolutionTagHelper = CreateTagHelper<global::Microsoft.AspNetCore.Mvc.Razor.TagHelpers.UrlResolutionTagHelper>();
                __tagHelperExecutionContext.Add(__Microsoft_AspNetCore_Mvc_Razor_TagHelpers_UrlResolutionTagHelper);
                __tagHelperExecutionContext.AddHtmlAttribute(__tagHelperAttribute_14);
                await __tagHelperRunner.RunAsync(__tagHelperExecutionContext);
                if (!__tagHelperExecutionContext.Output.IsContentModified)
                {
                    await __tagHelperExecutionContext.SetOutputContentAsync();
                }
                Write(__tagHelperExecutionContext.Output);
                __tagHelperExecutionContext = __tagHelperScopeManager.End();
                WriteLiteral("\r\n");
            }
            );
            WriteLiteral("\r\n");
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
        public global::Microsoft.AspNetCore.Mvc.Rendering.IHtmlHelper<ReturnViewModel> Html { get; private set; }
    }
}
#pragma warning restore 1591