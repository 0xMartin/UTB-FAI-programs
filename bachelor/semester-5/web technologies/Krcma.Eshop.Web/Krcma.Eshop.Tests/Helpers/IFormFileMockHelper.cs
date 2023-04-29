using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

using Microsoft.AspNetCore.Http;

using Xunit.Abstractions;
using Moq;

namespace Vogeltanz.Eshop.Tests.Helpers
{
    public class IFormFileMockHelper
    {
        private readonly ITestOutputHelper _testOutputHelper;
        public IFormFileMockHelper(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        public Mock<IFormFile> MockIFormFile(MemoryStream ms, StreamWriter writer, string fileName, string content, string contentType)
        {
            //IFormFile iff = new FormFile(...);
            Mock<IFormFile> iffMock = new Mock<IFormFile>();


            //vytvoření fake souboru do paměti použitím memory stream
            writer.Write(content);
            writer.Flush();
            ms.Position = 0;


            //nastavení fakeové IFormFile 
            //iffMock.Setup(iff => iff.OpenReadStream()).Returns(ms);
            iffMock.Setup(iff => iff.FileName).Returns(fileName);
            iffMock.Setup(iff => iff.ContentType).Returns(contentType);
            iffMock.Setup(iff => iff.Length).Returns(ms.Length);
            iffMock.Setup(iff => iff.CopyToAsync(It.IsAny<Stream>(), CancellationToken.None))
                                    .Callback<Stream, CancellationToken>((stream, token) =>
                                    {
                                        //pokud se zavolá metoda CopyToAsync, tak se provede tato část
                                        //kde se volá CopyTo v MemoryStream
                                        ms.CopyTo(stream);
                                    })
                                    .Returns(Task.CompletedTask);


            _testOutputHelper.WriteLine("MockIFormFile() -> ms.Length: " + ms.Length);

            return iffMock;
        }
    }
}
