using System;
using Microsoft.EntityFrameworkCore.Metadata;
using Microsoft.EntityFrameworkCore.Migrations;

namespace Krcma.Eshop.Web.Migrations.MySql
{
    public partial class My_Sql_115 : Migration
    {
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "ShippingTime",
                table: "Order");

            migrationBuilder.AddColumn<int>(
                name: "ShippingDelay",
                table: "Order",
                type: "int",
                nullable: false,
                defaultValue: 0);
        }

        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "ShippingDelay",
                table: "Order");

            migrationBuilder.AddColumn<DateTime>(
                name: "ShippingTime",
                table: "Order",
                type: "datetime(6)",
                nullable: false)
                .Annotation("MySql:ValueGenerationStrategy", MySqlValueGenerationStrategy.ComputedColumn);
        }
    }
}
