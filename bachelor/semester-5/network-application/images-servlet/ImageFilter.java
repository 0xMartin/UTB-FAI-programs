import java.util.List;

import javax.imageio.ImageIO;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.apache.commons.fileupload.FileItem;
import org.apache.commons.fileupload.FileUploadException;
import org.apache.commons.io.IOUtils;

import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

public class ImageFilter {

	public static final String DEFAULT_FILE_NAME = "file";

	private final String name;
	private final Method filterFunc;
	private final List<ImageFilter.Input> inputs;

	private BufferedImage inputImage;
	private String inputImage_type;
	private BufferedInputStream inputImage_stream;

	private BufferedImage outputImage;
	private BufferedInputStream outputImage_stream;

	/**
	 * Vytvori instanci tridy pro nastavovani obrazoveho filtru fitru
	 * 
	 * @param filterFunc - funkce ktera se vola v okamziku kdy ma byt aplikovany
	 *                   filtr na vstupni obrazek
	 * @param inputs     - form inputs (filter intputs)
	 */
	public ImageFilter(String name, Method filterFunc, List<ImageFilter.Input> inputs) {
		this.name = name;
		this.filterFunc = filterFunc;
		this.inputs = inputs;
	}

	/**
	 * Vygeneruje html code ktery bude obsahovat form inputy pro nastavovani filtru
	 * 
	 * @return HTML string
	 */
	public final String getHTMLCode() {
		final StringBuilder html = new StringBuilder();
		html.append("<h2>" + this.name + "</h2>");
		html.append("<form  method='post' enctype='multipart/form-data' class='border-bottom'>");
		html.append("<div class=\"d-flex justify-content-around pb-3\">");

		for (ImageFilter.Input i : this.inputs) {
			html.append("<div class=\"d-flex align-items-center\">");
			html.append("<label class='font-weight-bold text-uppercase px-3'>" + i.label + "</label>");
			html.append("</div>");
			switch (i.type) {
			case FILE:
				html.append("<input type=\"file\" class=\"form-control-file\" name='" + i.name + "'/>");
				break;
			case NUMBER:
				Object[] vals = (Object[]) i.value;
				html.append("<input type=\"number\" class=\"form-control-file\" name='" + i.name + "' value='" + vals[0].toString()
						+ "' min='"+vals[1].toString()+"' max='"+vals[2].toString()+"' step='"+vals[3].toString()+"'/>");
				break;
			case SELECT:
				html.append("<select style=\"width:100%\" name='" + i.name + "'>");
				Object[] opts = (Object[]) i.value;
				for (int j = 0; j < opts.length; ++j) {
					if (j == 0) {
						html.append("<option value='" + opts[j].toString() + "' selected>" + opts[j].toString()
								+ "</option>");
					} else {
						html.append("<option value='" + opts[j].toString() + "'>" + opts[j].toString() + "</option>");
					}
				}
				html.append("</select>");
				break;
			}
		}
		html.append("<div class='pl-2'>");
		html.append("<input type=\"hidden\" name='filter-name' value='" + this.name + "'/>");
		html.append("<button type=\"submit\" class=\"btn btn-primary\">Apply</button>");
		html.append("</div>");
		html.append("</div>");
		html.append("</form>");
		return html.toString();
	}

	/**
	 * @return Vstupy pro nastavovani filtru
	 */
	public final List<ImageFilter.Input> getInputs() {
		return this.inputs;
	}

	/**
	 * Nacte vstupni obrazek
	 * 
	 * @param inputStream - ImageInputStream
	 * @throws IOException
	 */
	public final void loadInputImage(final InputStream inputStream, long size, String type) throws IOException {
		this.inputImage_type = type;
		this.inputImage_stream = new BufferedInputStream(inputStream);
		this.inputImage_stream.mark((int) size + 1);
		this.inputImage = ImageIO.read(this.inputImage_stream);
	}

	/**
	 * Navrati vstupni obrazek
	 * 
	 * @return BufferedImage
	 */
	public BufferedImage getInputImage() {
		return this.inputImage;
	}

	/**
	 * Nastavy vystupni obrazek
	 * 
	 * @param outputImage - BufferedImage
	 */
	public void setOutputImage(BufferedImage outputImage) {
		this.outputImage = outputImage;

		try {
			ByteArrayOutputStream os = new ByteArrayOutputStream();
			ImageIO.write(outputImage, "jpeg", os);
			InputStream is = new ByteArrayInputStream(os.toByteArray());
			int size = is.available() + 1;
			this.outputImage_stream = new BufferedInputStream(is);
			this.outputImage_stream.mark(size);
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	/**
	 * Navrati vystupni obrazek
	 * 
	 * @return BufferedImage
	 */
	public BufferedImage getOutputImage() {
		return this.outputImage;
	}

	public String getName() {
		return this.name;
	}

	public boolean getImage(HttpServletRequest request, HttpServletResponse response) throws IOException {
		// zkontorluje jmeno filtru
		String param = request.getParameter("filter_name");
		if(param == null) return false;
		if (!param.endsWith(this.name.replaceAll("\\s+", "_"))) {
			return false;
		}

		param = request.getParameter("get_image");
		if (param != null) {
			OutputStream out = response.getOutputStream();
			if (param.endsWith("input") && this.inputImage_stream != null) {
				response.setContentType(this.inputImage_type);
				this.inputImage_stream.reset();
				IOUtils.copy(this.inputImage_stream, out);
			} else if (param.endsWith("output") && this.outputImage_stream != null) {
				response.setContentType("image/jpeg");
				this.outputImage_stream.reset();
				IOUtils.copy(this.outputImage_stream, out);
			}
			out.close();
			return true;
		}
		return false;
	}

	/**
	 * Zpracuje vstupni data
	 * 
	 * @throws ImageFilter.ImageFilterException
	 * @throws InvocationTargetException
	 * @throws IllegalArgumentException
	 * @throws IllegalAccessException
	 * @throws FileUploadException
	 * @throws IOException
	 */
	public final boolean proccessInputData(List<FileItem> items) throws Exception {
		if (this.filterFunc == null)
			throw new ImageFilter.ImageFilterException("Filter function for " + this.toString() + " is null");

		// overi typ filtru
		FileItem item = items.stream().filter(i -> "filter-name".equals(i.getFieldName())).findAny().orElse(null);
		if (item == null)
			return false;
		if (!item.getString().endsWith(this.name))
			return false;

		// nacte vstupni obrazek (default)
		item = items.stream().filter(i -> ImageFilter.DEFAULT_FILE_NAME.equals(i.getFieldName())).findAny()
				.orElse(null);
		if (item != null) {
			if (!item.getName().isEmpty() && item.getSize() > 0)
				this.loadInputImage(item.getInputStream(), item.getSize(), item.getContentType());
		}

		if (this.inputImage == null)
			throw new ImageFilter.ImageFilterException("Input Image for " + this.toString() + " is null");

		// zavola metodu pro dany filtr (nastaveni paramteru pro dany filtr + aplikovani
		// filtru)
		this.filterFunc.invoke(null, this, items);
		
		return true;
	}

	/**
	 * Podporovane input filtery
	 */
	public enum InputType {
		FILE, 
		NUMBER,	// Input.value: currenet_value, min, max, step
		SELECT	// Input.value: option1(selected), option2, ...
	}

	/**
	 * Form input pro nastavovani parametru obrazkoveho filtru
	 */
	public static class Input {
		public String label;
		public String name;
		public ImageFilter.InputType type;
		public Object value;

		public Input(String label, String name, ImageFilter.InputType type, Object value) {
			this.label = label;
			this.name = name;
			this.type = type;
			this.value = value;
		}

	}

	/**
	 * Vyjimka pro tridu ImageFilter
	 */
	public static class ImageFilterException extends Exception {
		private static final long serialVersionUID = 1L;

		public ImageFilterException(String errorMessage) {
			super(errorMessage);
		}
	}

}
