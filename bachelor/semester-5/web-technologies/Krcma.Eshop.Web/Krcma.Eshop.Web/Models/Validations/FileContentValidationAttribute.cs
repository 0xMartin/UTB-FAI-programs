using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc.ModelBinding.Validation;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Threading.Tasks;

namespace Krcma.Eshop.Web.Models.Validations
{
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field | AttributeTargets.Parameter)]
    public class FileContentValidationAttribute : ValidationAttribute, IClientModelValidator
    {
        public string ContentType { get; set; }
        
        public FileContentValidationAttribute(string ContentType)
        {
            this.ContentType = ContentType;
        }

        protected override ValidationResult IsValid(object value, ValidationContext validationContext)
        {
            if(value == null)
            {
                return ValidationResult.Success;
            }
            else if(value is IFormFile formFile)
            {
                if(formFile.ContentType.ToLower().Contains(this.ContentType.ToLower()))
                {
                    return ValidationResult.Success;
                }
                else
                {
                    return new ValidationResult($"Content of the file {validationContext.MemberName} is not {ContentType} .");
                }
            }


            throw new NotImplementedException($"This attribute {nameof(FileContentValidationAttribute)} is not implemented yet");
        }

        public void AddValidation(ClientModelValidationContext context)
        {
            context.Attributes.Add("data-val", "true");
            context.Attributes.Add("data-val-filecontent", $"Content of the file is not {ContentType} .");
            context.Attributes.Add("data-val-filecontent-type", ContentType.ToLower());

        }
    }
}
