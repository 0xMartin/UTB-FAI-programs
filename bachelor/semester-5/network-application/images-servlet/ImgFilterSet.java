import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.fileupload.FileItem;

import com.jhlabs.image.CircleFilter;
import com.jhlabs.image.CrystallizeFilter;
import com.jhlabs.image.GaussianFilter;

public class ImgFilterSet {

	public static final void filter1(final ImageFilter imageFilter, final List<FileItem> items) {
		FileItem item = items.stream().filter(i -> "radius".equals(i.getFieldName())).findAny().orElse(null);
		if(item != null) {
			GaussianFilter filter = new GaussianFilter(Integer.parseInt(item.getString()));
			imageFilter.setOutputImage(filter.filter(imageFilter.getInputImage(), null));
		}
	}
	
	public static final void filter2(final ImageFilter imageFilter, final List<FileItem> items) {
		FileItem item = items.stream().filter(i -> "thickness".equals(i.getFieldName())).findAny().orElse(null);
		if(item != null) {
			CrystallizeFilter filter = new CrystallizeFilter();
			filter.setEdgeThickness(Float.parseFloat(item.getString()));
			imageFilter.setOutputImage(filter.filter(imageFilter.getInputImage(), null));
		}
	}
	
	public static final void filter3(final ImageFilter imageFilter, final List<FileItem> items) {		
		CircleFilter filter = new CircleFilter();
		
		FileItem item = items.stream().filter(i -> "height".equals(i.getFieldName())).findAny().orElse(null);
		if(item != null) {
			filter.setHeight(Float.parseFloat(item.getString()));
		}
		item = items.stream().filter(i -> "x".equals(i.getFieldName())).findAny().orElse(null);
		if(item != null) {
			filter.setCentreX(Float.parseFloat(item.getString()));
		}
		item = items.stream().filter(i -> "y".equals(i.getFieldName())).findAny().orElse(null);
		if(item != null) {
			filter.setCentreY(Float.parseFloat(item.getString()));
		}
		
		imageFilter.setOutputImage(filter.filter(imageFilter.getInputImage(), null));
	}

	/**
	 * Vytvoreni sady filtru
	 * @return List<ImageFilter>
	 * @throws NoSuchMethodException
	 * @throws SecurityException
	 */
	public static final List<ImageFilter> createImageFilters() throws NoSuchMethodException, SecurityException {
		final List<ImageFilter> filters = new ArrayList<ImageFilter>();
		List<ImageFilter.Input> inputs;

		// filter 1 (GaussianFilter)
		/**************************************************************************************************/
		inputs = Arrays.asList(
				new ImageFilter.Input(
						"Input image", 
						ImageFilter.DEFAULT_FILE_NAME, 
						ImageFilter.InputType.FILE, 
						null),
				new ImageFilter.Input(
						"Blur radius", 
						"radius", 
						ImageFilter.InputType.SELECT, 
						new Object[] {1, 3, 5, 10, 25, 50, 100, 250})
				);
		filters.add(new ImageFilter("BLUR FILTER", ImgFilterSet.class.getMethod("filter1", ImageFilter.class, List.class), inputs));
		
		// filter 2 (CrystallizeFilter)
		/**************************************************************************************************/
		inputs = Arrays.asList(
				new ImageFilter.Input(
						"Input image", 
						ImageFilter.DEFAULT_FILE_NAME, 
						ImageFilter.InputType.FILE,
						null),
				new ImageFilter.Input(
						"Edge Thickness", 
						"thickness", 
						ImageFilter.InputType.NUMBER, 
						new Object[] {0.4, 0.05, 1000.0, 0.05})
				);
		filters.add(new ImageFilter("CRYSTALLIZE FILTER", ImgFilterSet.class.getMethod("filter2", ImageFilter.class, List.class), inputs));
		
		// filter 3 (RaysFilter)
		/**************************************************************************************************/
		inputs = Arrays.asList(
				new ImageFilter.Input(
						"Input image", 
						ImageFilter.DEFAULT_FILE_NAME,
						ImageFilter.InputType.FILE,
						null),
				new ImageFilter.Input(
						"Height", 
						"height", 
						ImageFilter.InputType.NUMBER,
						new Object[] {200.0, 10.0, 2000.0, 1.0}),
				new ImageFilter.Input(
						"X", 
						"x", 
						ImageFilter.InputType.NUMBER, 
						new Object[] {0.5, 0.0, 1.0, 0.01}),
				new ImageFilter.Input(
						"Y", 
						"y", 
						ImageFilter.InputType.NUMBER,
						new Object[] {0.7, 0.0, 1.0, 0.01})
				);
		filters.add(new ImageFilter("CIRCLE FILTER", ImgFilterSet.class.getMethod("filter3", ImageFilter.class, List.class), inputs));

		return filters;
	}

}
