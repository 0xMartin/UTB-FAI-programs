using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

using Microsoft.AspNetCore.Http;

using Xunit;
using Xunit.Abstractions;
using Moq;

using Krcma.Eshop.Web.Models.Implementation;

using Vogeltanz.Eshop.Tests.Helpers;

namespace Krcma.Eshop.Tests
{
    public class FileUploadTests
    {
        const string relativeDirectoryPath = "/images";


        private readonly ITestOutputHelper _testOutputHelper;
        public FileUploadTests(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        [Fact]
        public void CreateOutputDirectoryForUploadTest_Success()
        {
            DirectoryInfo directoryInfo = Directory.CreateDirectory(Path.Combine(Directory.GetCurrentDirectory() + relativeDirectoryPath));

            Assert.NotNull(directoryInfo);

            _testOutputHelper.WriteLine(directoryInfo.FullName);
            //Directory.Delete(directoryInfo.FullName, true);
        }


        [Fact]
        public async Task UploadImageFile_Success()
        {

            //Arrange
            string content = "‰PNG" + "FakeImageContent";
            string fileName = "UploadImageFile.png";

            string ImageSrc = String.Empty;

            Directory.CreateDirectory(Path.Combine(Directory.GetCurrentDirectory() + relativeDirectoryPath));

            string relativePath = relativeDirectoryPath + "\\" + fileName;
            string path = Directory.GetCurrentDirectory() + relativePath;

            //nastavení fakeové IFormFile pomocí MemoryStream
            using (var ms = new MemoryStream())
            {
                using (var writer = new StreamWriter(ms))
                {
                    IFormFileMockHelper iffMockHelper = new IFormFileMockHelper(_testOutputHelper);
                    Mock<IFormFile> iffMock = iffMockHelper.MockIFormFile(ms, writer, fileName, content, "image/png");


                    FileUpload fup = new FileUpload(Directory.GetCurrentDirectory(), "images", "image");

                    //Act
                    ImageSrc = await fup.FileUploadAsync(iffMock.Object);
                }
            }



            //Assert
            Assert.False(String.IsNullOrWhiteSpace(ImageSrc));

            _testOutputHelper.WriteLine("vytvoøená cesta path: " + path);
            _testOutputHelper.WriteLine("vytvoøená cesta relativePath: " + relativePath);
            _testOutputHelper.WriteLine("vytvoøená cesta ImageSrc: " + ImageSrc);
            Assert.True(ImageSrc == relativePath);

            Assert.True(File.Exists(path));

            File.Delete(path);
            Assert.False(File.Exists(path));
        }


        [Fact]
        public async Task UploadTextFile_Failure()
        {

            //Arrange
            string content = "Fake text content";
            string fileName = "UploadTextFile.txt";

            string ImageSrc = String.Empty;

            Directory.CreateDirectory(Path.Combine(Directory.GetCurrentDirectory() + relativeDirectoryPath));

            string relativePath = relativeDirectoryPath + "\\" + fileName;
            string path = Directory.GetCurrentDirectory() + relativePath;

            //nastavení fakeové IFormFile pomocí MemoryStream
            using (var ms = new MemoryStream())
            {
                using (var writer = new StreamWriter(ms))
                {
                    IFormFileMockHelper iffMockHelper = new IFormFileMockHelper(_testOutputHelper);
                    Mock<IFormFile> iffMock = iffMockHelper.MockIFormFile(ms, writer, fileName, content, "text/txt");


                    FileUpload fup = new FileUpload(Directory.GetCurrentDirectory(), "Carousels", "image");

                    //Act
                    ImageSrc = await fup.FileUploadAsync(iffMock.Object);
                }
            }



            //Assert
            Assert.True(String.IsNullOrWhiteSpace(ImageSrc));

            _testOutputHelper.WriteLine("vytvoøená cesta path: " + path);
            _testOutputHelper.WriteLine("vytvoøená cesta relativePath: " + relativePath);
            _testOutputHelper.WriteLine("vytvoøená cesta ImageSrc: " + ImageSrc);
            Assert.False(ImageSrc == relativePath);

            Assert.False(File.Exists(path));

            File.Delete(path);
        }


        

    }
}
