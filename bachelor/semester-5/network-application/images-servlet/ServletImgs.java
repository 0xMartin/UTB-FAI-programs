/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

import java.io.*;
import java.util.List;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

import org.apache.commons.fileupload.FileItem;
import org.apache.commons.fileupload.FileItemFactory;
import org.apache.commons.fileupload.FileUploadException;
import org.apache.commons.fileupload.disk.DiskFileItemFactory;
import org.apache.commons.fileupload.servlet.ServletFileUpload;

/**
 *
 * @author UserXP
 */
@WebServlet("/Filters")
public class ServletImgs extends HttpServlet {

	private static final long serialVersionUID = 1L;

	public static final String KEY_LAST_USED_FILTER = "last_used_filter";
	public static final String KEY_FILTER_SET = "filters";

	protected void printPage(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		response.setContentType("text/html;charset=UTF-8");

		final PrintWriter out = response.getWriter();
		final List<ImageFilter> filters = getImageFilterSet(request.getSession());

		try {
			out.println("<html>");
			out.println("<head>");
			out.println("<title>Image filters</title>");
			out.println(
					"<link rel=\"stylesheet\" href=\"https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css\" >");
			out.println("</head>");
			out.println("<body>");
			out.println("<div class=\"container p-5\">");

			if (filters != null) {
				ImageFilter selected_filter = (ImageFilter) request.getSession()
						.getAttribute(ServletImgs.KEY_LAST_USED_FILTER);
				if (selected_filter == null)
					selected_filter = filters.get(0);

				// tab header
				out.println("<ul class=\"nav nav-tabs\" id=\"myTab\" role=\"tablist\">");
				for (ImageFilter f : filters) {
					out.println("<li class=\"nav-item\" role=\"presentation\">");
					if (f == selected_filter) {
						out.println("<button class=\"nav-link active\" data-bs-toggle=\"tab\" data-bs-target=\"#"
								+ f.getName().replaceAll("\\s+", "_")
								+ "\" type=\"button\" role=\"tab\" aria-controls=\"home\" aria-selected=\"true\">"
								+ f.getName() + "</button>");
					} else {
						out.println("<button class=\"nav-link\" data-bs-toggle=\"tab\" data-bs-target=\"#"
								+ f.getName().replaceAll("\\s+", "_")
								+ "\" type=\"button\" role=\"tab\" aria-controls=\"profile\" aria-selected=\"false\">"
								+ f.getName() + "</button>\r\n" + "  </li>");
					}
					out.println("</li>");
				}
				out.println("</ul>");

				// tabs
				out.println("<div class=\"tab-content\" id=\"myTabContent\">");
				for (ImageFilter f : filters) {
					if (f == selected_filter) {
						out.println("<div class=\"tab-pane fade show active\" id=\""
								+ f.getName().replaceAll("\\s+", "_") + "\" role=\"tabpanel\">");
					} else {
						out.println("<div class=\"tab-pane fade show\" id=\"" + f.getName().replaceAll("\\s+", "_")
								+ "\" role=\"tabpanel\">");
					}

					// form
					out.println(f.getHTMLCode());
					// vystupni obrazky
					out.println("<div class=\"d-flex justify-content-around\">");
					if (f.getInputImage() != null) {
						out.println("<div class='col-sm'>");
						out.println("<img style='width: 30vw' src='" + request.getRequestURI()
								+ "?get_image=input&filter_name=" + f.getName().replaceAll("\\s+", "_") + "'>");
						out.println("</div>");
					}
					if (f.getOutputImage() != null) {
						out.println("<div class='col-sm'>");
						out.println("<img style='width: 30vw' src='" + request.getRequestURI()
								+ "?get_image=output&filter_name=" + f.getName().replaceAll("\\s+", "_") + "'>");
						out.println("</div>");
					}
					out.println("</div>");

					out.println("</div>");
				}
				out.println("</div>");
			}

			out.println("</div>");
			out.println("</body>");
			out.println(
					"<script src=\"https://cdn.jsdelivr.net/npm/bootstrap@5.1.2/dist/js/bootstrap.bundle.min.js\"></script>");
			out.println("</html>");
		} finally {
			out.close();
		}
	}

	@SuppressWarnings("unchecked")
	public List<ImageFilter> getImageFilterSet(HttpSession session) {
		List<ImageFilter> filterSet = (List<ImageFilter>) session.getAttribute(ServletImgs.KEY_FILTER_SET);
		if (filterSet == null) {
			try {
				filterSet = ImgFilterSet.createImageFilters();
				session.setAttribute("filters", filterSet);
			} catch (NoSuchMethodException | SecurityException e) {
				e.printStackTrace();
			}
		}
		return filterSet;
	}

	@Override
	protected void doGet(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {

		List<ImageFilter> filters = getImageFilterSet(request.getSession());

		if (filters != null) {
			for (ImageFilter f : filters) {
				try {
					if (f.getImage(request, response))
						return;
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}

		printPage(request, response);
	}

	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {

		List<ImageFilter> filters = getImageFilterSet(request.getSession());
		if (filters != null) {
			if (ServletFileUpload.isMultipartContent(request)) {
				FileItemFactory factory = new DiskFileItemFactory();
				ServletFileUpload fileupload = new ServletFileUpload(factory);
				try {
					List<FileItem> items = fileupload.parseRequest(request);

					for (ImageFilter f : filters) {
						try {
							if (f.proccessInputData(items)) {
								request.getSession().setAttribute(ServletImgs.KEY_LAST_USED_FILTER, f);
								break;
							}
						} catch (Exception e) {
							e.printStackTrace();
						}
					}
				} catch (FileUploadException e1) {
					e1.printStackTrace();
				}
			}
		}

		printPage(request, response);
	}

	@Override
	public String getServletInfo() {
		return "Servlet pro zpracovani obrazku";
	}

}
