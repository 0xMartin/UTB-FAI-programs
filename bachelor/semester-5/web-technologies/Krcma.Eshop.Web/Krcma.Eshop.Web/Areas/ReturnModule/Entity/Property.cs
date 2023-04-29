using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Threading.Tasks;

namespace Krcma.Eshop.Web.Areas.ReturnModule.Entity
{
    [Table(nameof(Property))]
    public class Property
    {

        [Key]
        [Required]
        public string Name { get; set; }

        [Required]
        public string Value { get; set; }

    }
}
