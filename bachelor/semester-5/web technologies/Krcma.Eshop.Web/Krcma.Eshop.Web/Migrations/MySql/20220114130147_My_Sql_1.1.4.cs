using Microsoft.EntityFrameworkCore.Migrations;

namespace Krcma.Eshop.Web.Migrations.MySql
{
    public partial class My_Sql_114 : Migration
    {
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateIndex(
                name: "IX_ReturnProduct_OrderItemID",
                table: "ReturnProduct",
                column: "OrderItemID");

            migrationBuilder.AddForeignKey(
                name: "FK_ReturnProduct_OrderItem_OrderItemID",
                table: "ReturnProduct",
                column: "OrderItemID",
                principalTable: "OrderItem",
                principalColumn: "ID",
                onDelete: ReferentialAction.Cascade);
        }

        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_ReturnProduct_OrderItem_OrderItemID",
                table: "ReturnProduct");

            migrationBuilder.DropIndex(
                name: "IX_ReturnProduct_OrderItemID",
                table: "ReturnProduct");
        }
    }
}
